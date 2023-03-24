from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from os import path as osp
import os
from scipy.signal import savgol_filter
from scipy.ndimage.interpolation import shift
from concurrent.futures import ThreadPoolExecutor
import traceback

from skeleton_tools.openpose_layouts.body import COCO_LAYOUT
from skeleton_tools.openpose_layouts.face import PYFEAT_FACIAL
from skeleton_tools.skeleton_visualization.data_prepare.data_extract import MMPoseDataExtractor, PyfeatDataExtractor
from skeleton_tools.skeleton_visualization.data_prepare.writer import VideoWriter, ImageWriter
from skeleton_tools.skeleton_visualization.painters.base_painters import GlobalPainter
from skeleton_tools.skeleton_visualization.painters.dynamic_graphs import DynamicPolar, DynamicSignal, DynamicSkeleton
from skeleton_tools.skeleton_visualization.painters.local_painters import GraphPainter, LabelPainter, ScorePainter, BoxPainter
from skeleton_tools.utils.constants import AU_COLS, EMOTION_COLS
from skeleton_tools.utils.tools import read_pkl


class VideoCreator:
    def __init__(self, global_painters=(), local_painters=(), graphs=()):
        self.global_painters = global_painters
        self.local_painters = local_painters
        self.graphs = graphs

    def process_frame(self, frame, video_data, i):
        M, T = video_data['landmarks'].shape[:2]
        if i >= T:
            return frame
        paint_frame = frame.copy()
        try:
            for painter in self.global_painters:
                paint_frame = painter(paint_frame, video_data, i)
            for painter in self.local_painters:
                for j in range(M):
                    paint_frame = painter(paint_frame, video_data, i, j)
            if any(self.graphs):
                graphs = [graph(i) for graph in self.graphs]
                paint_frame = np.concatenate((paint_frame, np.concatenate(graphs, axis=0)), axis=1)
        except Exception as e:
            print(f'Error at frame {i}: {e}')
            print(traceback.format_exc())
            raise e
        return paint_frame


    def _create(self, video_path, video_data, writer, start=None, end=None, unit='frame'):
        fps, frame_count, length = video_data['fps'], video_data['frame_count'], video_data['duration_seconds']
        start, end = int(0 if start is None else start), int((length if unit == 'time' else frame_count) if end is None else end)
        if unit == 'time':
            start, end = int(start * fps), int(end * fps)
        cap = cv2.VideoCapture(video_path)
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        # for i in tqdm(range(start, end), desc="Writing video result"):
        #     ret, frame = cap.read()
        #     if not ret:
        #         continue
        #     paint_frame = self.process_frame(frame, video_data, i)
        #     writer.write(paint_frame)
        batch_size = 256
        with ThreadPoolExecutor(max_workers=32) as p:
            for i in tqdm(range(start, end, batch_size), desc="Writing video result"):
                frames = [cap.read() for _ in range(batch_size)]
                frames = [frame for ret, frame in frames if ret]
                frames = [p.submit(self.process_frame, frame, video_data, i + j) for j, frame in enumerate(frames)]
                for f in frames:
                    writer.write(f.result())
        cap.release()
        writer.release()

    def create_video(self, video_path, video_data, out_path, start=None, end=None, unit='frame'):
        fps, frame_count, length, (width, height) = video_data['fps'], video_data['frame_count'], video_data['duration_seconds'], video_data['resolution']
        writer = VideoWriter(out_path, fps, (width + self.graphs[0].width if any(self.graphs) else 0, height))
        self._create(video_path, video_data, writer, start, end, unit)

    def create_image(self, video_path, video_data, out_path, start=None, end=None, unit='frame'):
        writer = ImageWriter(out_path, start=start)
        self._create(video_path, video_data, writer, start, end, unit)


def interpolate(seq):
    n = seq.shape[1]
    df = pd.DataFrame(seq)
    mask = df.copy()
    grp = ((mask.notnull() != mask.shift().notnull()).cumsum())
    grp['ones'] = 1
    for col in df.columns:
        mask[col] = (grp.groupby(col)['ones'].transform('count') < 10) | df[col].notnull()
    interpolated = df.interpolate(limit_direction='both', limit_area='inside').bfill()[mask]
    interpolated['interpolated'] = df.apply(lambda row: 1 if row.isna().any() else 0, axis=1)
    return interpolated[[x for x in range(n)]].to_numpy()


def create_barni(video_path, pkl_path, out_path, start=None, end=None):
    _, ext = osp.splitext(video_path)
    extractor = PyfeatDataExtractor(PYFEAT_FACIAL)
    data = extractor(pkl_path)
    child_face_score = np.array([data['landmarks_scores'][c, i] for i, c in enumerate(data['child_ids'])]).T[0]
    child_aus = np.array([data['aus'][c, i] for i, c in enumerate(data['child_ids'])])
    child_emotions = np.array([data['emotions'][c, i] for i, c in enumerate(data['child_ids'])])

    t = 0.985
    child_aus[child_face_score < t] = np.nan
    child_emotions[child_face_score < t] = np.nan
    child_aus = interpolate(child_aus)
    child_emotions = interpolate(child_emotions)

    local_painters = [GraphPainter(extractor.graph_layout, epsilon=t, alpha=0.4), LabelPainter(), ScorePainter(), BoxPainter()]
    global_painters = [GlobalPainter(p) for p in local_painters]
    graphs = [DynamicPolar('AUs', child_aus, AU_COLS, int(data['resolution'][1] * 0.5), filters=()),
              DynamicSignal('Emotions', child_emotions, EMOTION_COLS,
                            'Time', 'Score',
                            window_size=1000,
                            height=int(data['resolution'][1] * 0.5),
                            filters=())]
    vc = VideoCreator(global_painters=global_painters, local_painters=[], graphs=graphs)
    vc.create_video(video_path, data, out_path, start=start, end=end, unit='time')


def create_jordi(video_path, skeleton_path, predictions_path, out_path, start=None, end=None):
    extractor = MMPoseDataExtractor(COCO_LAYOUT)
    data = extractor(skeleton_path, predictions_path)
    width, height = data['resolution']
    epsilon = 0.3
    local_painters = [GraphPainter(extractor.graph_layout, epsilon=epsilon, alpha=0.4), LabelPainter(), ScorePainter(), BoxPainter()]
    global_painters = [GlobalPainter(p) for p in local_painters]
    graphs = [DynamicSignal('Stereotypical Action', data['predictions'], ['Stereotypical'],
                            'Time', 'Score',
                            window_size=1000,
                            height=int(height * 0.5),
                            width=width,
                            filters=()),
              DynamicSkeleton(layout=extractor.graph_layout, epsilon=epsilon, data=data,
                              height=int(height * 0.5), width=width)]

    vc = VideoCreator(global_painters=global_painters, local_painters=[], graphs=graphs)
    vc.create_video(video_path, data, out_path, start=start, end=end, unit='frame')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-video", "--video_path")
    parser.add_argument("-pkl", "--pkl_path")
    parser.add_argument("-pkl_pred", "--pkl_predictions_path")
    parser.add_argument("-out", "--out_dir")
    parser.add_argument("-start", "--start")
    parser.add_argument("-end", "--end")
    parser.add_argument('-j', '--jordi', action='store_true')
    parser.add_argument('-b', '--barni', action='store_true')
    args = parser.parse_args()

    name, ext = osp.splitext(osp.basename(args.video_path))
    if args.jordi:
        create_jordi(args.video_path, args.pkl_path, args.pkl_predictions_path, osp.join(args.out_dir, f'{name}_SMMs_{args.start}_{args.end}.mp4'), args.start, args.end)
    if args.barni:
        create_barni(args.video_path, args.pkl_path, osp.join(args.out_dir, f'{name}_Facial_{args.start}_{args.end}.mp4'), args.start, args.end)

    # for v in ['Tamar_dinstein_cognitive',
    #           '1032323269_ADOS_Control_050123_1625_1',
    #           '673168102_ADOS_Clinical_040221_0926_2',
    #           '1032482908_ADOS_Clinical_310122_1057_1',
    #           '675793378_ADOS_Clinical_051021_0839_4',
    #           '1020232399_ADOS_310718_0922_4_Trim',
    #           '666808807_ADOS_Clinical_120218_1300_4',
    #           '675883867_ADOS_Clinical_150920_1249_2_Trim']:
    #     try:
    #         create_barni(v)
    #         break
    #     except Exception as e:
    #         print(f"Failed to create video for {v}: {e}")
    #         print(traceback.format_exc())

    # create_barni('673950985_ADOS_ASD_061019_1038_4_Trim', start=1127, end=1155)
    # create_barni('673950985_ADOS_ASD_061019_1038_4_Trim', start=1058, end=1076)
    # create_jordi('666770197_Cognitive_Clinical_010120_1127_3')
    # create_jordi('1007196724_ADOS_Clinical_190917_0000_2')

    # create_jordi('1018226485_ADOS_Control_210119_0929_4') # Tamar

    # video_name = r'666770197_Cognitive_Clinical_010120_1127_3'
    # jordi_dir = r'Z:\Users\TalBarami\jordi_cross_validation'
    # out_dir = r'Z:\Users\TalBarami\vids'
    # extractor = MMPoseDataExtractor(COCO_LAYOUT)
    # model = [m for m in os.listdir(osp.join(jordi_dir, video_name, 'jordi')) if osp.isdir(osp.join(jordi_dir, video_name, 'jordi', m))][0]
    # data = extractor(osp.join(jordi_dir, video_name, 'jordi', f'{video_name}.pkl'),
    #                  osp.join(jordi_dir, video_name, 'jordi', model, f'{video_name}_predictions.pkl'))
    # video_path = data['video_path']
    # predictions = data['predictions']
    # cap = cv2.VideoCapture(data['video_path'])
    # # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # gp = GraphPainter(extractor.graph_layout)
    # i = 0
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     for p in range(5):
    #         frame = gp(frame, data, i, p)
    #     i += 1
    #     cv2.imshow('frame', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break