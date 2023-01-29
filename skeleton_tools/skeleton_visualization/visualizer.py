import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from os import path as osp

from skeleton_tools.openpose_layouts.body import COCO_LAYOUT
from skeleton_tools.openpose_layouts.face import PYFEAT_FACIAL
from skeleton_tools.skeleton_visualization.data_prepare.data_extract import MMPoseDataExtractor, PyfeatDataExtractor
from skeleton_tools.skeleton_visualization.painters.dynamic_graphs import DynamicPolar, DynamicSignal
from skeleton_tools.skeleton_visualization.painters.painter import GraphPainter, LabelPainter, BoxPainter, BlurPainter
from skeleton_tools.utils.constants import AU_COLS, EMOTION_COLS
from skeleton_tools.utils.tools import read_pkl


class VideoCreator:
    def __init__(self, painters=(), graphs=()):
        self.painters = painters
        self.graphs = graphs

    def create_video(self, video_path, video_data, out_path, start=None, end=None, double_frame=False):
        fps, length, (width, height) = video_data['fps'], video_data['frame_count'], video_data['resolution']
        M = video_data['landmarks'].shape[0]
        start, end = int(0 if start is None else start), int(length if end is None else end)
        width_multiplier = 1 + int(double_frame)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (int(width_multiplier * width) + self.graphs[0].width if any(self.graphs) else 0, height))
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for i in tqdm(range(start, end), desc="Writing video result"):
            ret, frame = cap.read()
            paint_frame = np.zeros_like(frame) if double_frame else frame
            for j in range(M):
                for painter in self.painters:
                    paint_frame = painter(paint_frame, video_data, i, j)
            out_frame = np.concatenate((frame, paint_frame), axis=1) if double_frame else paint_frame
            if any(self.graphs):
                graphs = [graph(i) for graph in self.graphs]
                out_frame = np.concatenate((out_frame, np.concatenate(graphs, axis=0)), axis=1)
            out.write(out_frame)
        cap.release()
        out.release()

if __name__ == '__main__':
    vids_dir = r'Z:\Users\Liora\Project 1 - Video recorsings for analyses'
    barni_dir = r'Z:\Users\TalBarami\BARNI\skip_frames=1'
    out_dir = r'Z:\Users\TalBarami\BARNI\skip_frames=1\videos'
    file = '10212252882_ADOS_Clinical_151220_0925_2_Trim'
    extractor = PyfeatDataExtractor(PYFEAT_FACIAL)
    data = extractor(osp.join(barni_dir, file, 'barni', f'{file}.pkl'))
    child_aus = np.array([data['aus'][c, i] for i, c in enumerate(data['child_ids'])])
    child_emotions = np.array([data['emotions'][c, i] for i, c in enumerate(data['child_ids'])])
    painters = [BlurPainter(), GraphPainter(PYFEAT_FACIAL), LabelPainter(), BoxPainter()]
    graphs = [DynamicPolar('AUs', child_aus, AU_COLS, int(data['resolution'][1] * 0.5)),
              DynamicSignal('Emotions', child_emotions, EMOTION_COLS, 'Time', 'Score', np.round(np.max((200, data['landmarks'].shape[1] * 0.01)), -2).astype(int), int(data['resolution'][1] * 0.5))]
    vc = VideoCreator(painters, graphs)
    vc.create_video(osp.join(vids_dir, f'{file}.mp4'),
                    data,
                    osp.join(out_dir, f'{file}.mp4'), double_frame=False)
