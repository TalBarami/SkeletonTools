import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from os import path as osp
from scipy.signal import savgol_filter
import gc
from concurrent.futures import ThreadPoolExecutor

from skeleton_tools.openpose_layouts.body import COCO_LAYOUT
from skeleton_tools.openpose_layouts.face import PYFEAT_FACIAL
from skeleton_tools.skeleton_visualization.data_prepare.data_extract import MMPoseDataExtractor, PyfeatDataExtractor
from skeleton_tools.skeleton_visualization.painters.dynamic_graphs import DynamicPolar, DynamicSignal
from skeleton_tools.skeleton_visualization.painters.painter import GraphPainter, LabelPainter, BoxPainter, BlurPainter
from skeleton_tools.utils.constants import AU_COLS, EMOTION_COLS
from skeleton_tools.utils.tools import read_pkl


class VideoCreator:
    def __init__(self, painters=(), graphs=(), double_frame=False):
        self.painters = painters
        self.graphs = graphs
        self.double_frame = double_frame

    def process_frame(self, frame, video_data, i):
        paint_frame = np.zeros_like(frame) if self.double_frame else frame
        M = video_data['landmarks'].shape[0]
        for j in range(M):
            for painter in self.painters:
                paint_frame = painter(paint_frame, video_data, i, j)
        out_frame = np.concatenate((frame, paint_frame), axis=1) if self.double_frame else paint_frame
        if any(self.graphs):
            graphs = [graph(i) for graph in self.graphs]
            out_frame = np.concatenate((out_frame, np.concatenate(graphs, axis=0)), axis=1)
        return out_frame


    def create_video(self, video_path, video_data, out_path, start=None, end=None):
        fps, length, (width, height) = video_data['fps'], video_data['frame_count'], video_data['resolution']
        start, end = int(0 if start is None else start), int(length if end is None else end)
        width_multiplier = 1 + int(self.double_frame)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (int(width_multiplier * width) + self.graphs[0].width if any(self.graphs) else 0, height))
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        batch_size = 256
        with ThreadPoolExecutor(max_workers=32) as p:
            for i in tqdm(range(start, end, batch_size), desc="Writing video result"):
                frames = [cap.read() for _ in range(batch_size)]
                frames = [frame for ret, frame in frames if ret]
                frames = [p.submit(self.process_frame, frame, video_data, i + j) for j, frame in enumerate(frames)]
                for f in frames:
                    out.write(f.result())
        cap.release()
        out.release()


if __name__ == '__main__':
    vids_dir = r'Z:\Users\Liora\Project 1 - Video recorsings for analyses'
    barni_dir = r'Z:\Users\TalBarami\BARNI\skip_frames=1'
    out_dir = r'Z:\Users\TalBarami\BARNI\skip_frames=1\videos'
    file = '673950985_ADOS_ASD_061019_1038_4_Trim'
    extractor = PyfeatDataExtractor(PYFEAT_FACIAL)
    data = extractor(osp.join(barni_dir, file, 'barni', f'{file}.pkl'))
    child_face_score = np.array([data['landmarks_scores'][c, i] for i, c in enumerate(data['child_ids'])]).T[0]
    child_aus = np.array([data['aus'][c, i] for i, c in enumerate(data['child_ids'])])
    child_emotions = np.array([data['emotions'][c, i] for i, c in enumerate(data['child_ids'])])

    def interpolate(x):
        nans, y = np.isnan(x), lambda z: z.nonzero()[0]
        x[nans] = np.interp(y(nans), y(~nans), x[~nans])
        x = savgol_filter(x, 51, 3, axis=0)
        # x[child_face_score < 0.99] = np.nan
        return x

    painters = [BlurPainter(), GraphPainter(PYFEAT_FACIAL, epsilon=0.98), LabelPainter(), BoxPainter()]
    graphs = [DynamicPolar('AUs', interpolate(child_aus), AU_COLS, int(data['resolution'][1] * 0.5), filters=()),
              DynamicSignal('Emotions', interpolate(child_emotions), EMOTION_COLS,
                            'Time', 'Score',
                            window_size=np.round(np.max((200, data['landmarks'].shape[1] * 0.01)), -2).astype(int),
                            height=int(data['resolution'][1] * 0.5),
                            filters=())]
    vc = VideoCreator(painters, graphs, double_frame=False)
    for end in [1000, None]:
        vc.create_video(osp.join(vids_dir, f'{file}.mp4'),
                        data,
                        osp.join(out_dir, f'{file}2.mp4' if end is None else f'{file}_{end}.mp4'), end=end)