import traceback
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from os import path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

from omegaconf import OmegaConf
from tqdm import tqdm

from skeleton_tools.openpose_layouts.body import COCO_LAYOUT
from skeleton_tools.openpose_layouts.face import PYFEAT_FACIAL
from skeleton_tools.skeleton_visualization.components_loader import ComponentsLoader
from skeleton_tools.skeleton_visualization.data_prepare.data_extract import MMPoseDataExtractor, PyfeatDataExtractor
from skeleton_tools.skeleton_visualization.data_prepare.reader import VideoReader, DefaultReader
from skeleton_tools.skeleton_visualization.data_prepare.writer import VideoWriter, ImageWriter
from skeleton_tools.skeleton_visualization.paint_components.dynamic_graphs.dynamic_graphs import DynamicPolar, DynamicSignal, DynamicSkeleton
from skeleton_tools.skeleton_visualization.paint_components.frame_painters.base_painters import GlobalPainter, BlurPainter, ScaleAbsPainter
from skeleton_tools.skeleton_visualization.paint_components.frame_painters.local_painters import BoxPainter, GraphPainter, CustomTextPainter
from skeleton_tools.utils.tools import init_logger, load_config, init_directories, savgol_filter

plt.switch_backend('agg')


class VideoCreator:
    def __init__(self, global_painters=(), local_painters=(), graphs=(), scale=1):
        self.global_painters = global_painters
        self.local_painters = local_painters
        self.graphs = graphs
        self.scale = scale

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

    def _create(self, data, writer, video_path=None, start=None, end=None, unit='frame'):
        if video_path is None:
            reader = DefaultReader(resolution=data['resolution'], scale=self.scale)
        else:
            reader = VideoReader(video_path, scale=self.scale)
        fps, frame_count, length = data['fps'], data['frame_count'], data['duration_seconds']
        start, end = int(0 if start is None else start), int((length if unit == 'time' else frame_count) if end is None else end)
        if unit == 'time':
            start, end = int(start * fps), int(end * fps)
        i = 0
        while i < start:
            _, _ = reader.read()
            i += 1

        # for i in tqdm(range(start, end), desc="Writing video result"):
        #     ret, frame = cap.read()
        #     if not ret:
        #         continue
        #     paint_frame = self.process_frame(frame, video_data, i)
        #     writer.write(paint_frame)
        batch_size = 256
        with ThreadPoolExecutor(max_workers=32) as p:
            for i in tqdm(range(start, end, batch_size), desc="Writing video result"):
                frames = [reader.read() for _ in range(batch_size)]
                frames = [frame for ret, frame in frames if ret]
                frames = [p.submit(self.process_frame, frame, data, i + j) for j, frame in enumerate(frames)]
                for f in frames:
                    writer.write(f.result())
        reader.release()
        writer.release()

    def create_video(self, data, out_path, video_path=None, start=None, end=None, unit='frame'):
        out, ext = osp.splitext(out_path)
        fps, frame_count, length, (width, height) = data['fps'], data['frame_count'], data['duration_seconds'], data['resolution']
        writer = VideoWriter(out_path=f'{out}_{start}_{end}{ext}', fps=fps, resolution=(width + (self.graphs[0].width if any(self.graphs) else 0), height))
        self._create(video_path=video_path, data=data, writer=writer, start=start, end=end, unit=unit)

    def create_image(self, data, out_path, video_path=None, start=None, end=None, unit='frame'):
        writer = ImageWriter(f'{out_path}_{start}_{end}', start=start)
        self._create(video_path=video_path, data=data, writer=writer, start=start, end=end, unit=unit)


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

def create_barni(cfg, out_path, start=None, end=None, scale=1):
    extractor = PyfeatDataExtractor(PYFEAT_FACIAL, scale=scale)
    data = extractor(pkl_path)
    child_face_score = np.array([data['landmarks_scores'][c, i] for i, c in enumerate(data['child_ids'])]).T[0]
    child_aus = np.array([data['aus'][c, i] for i, c in enumerate(data['child_ids'])])
    child_emotions = np.array([data['emotions'][c, i] for i, c in enumerate(data['child_ids'])])

    t = 0.9
    child_aus[child_face_score < t] = np.nan
    child_emotions[child_face_score < t] = np.nan
    child_aus = interpolate(child_aus)
    child_emotions = interpolate(child_emotions)
    width, height = data['resolution']

    local_painters = [GraphPainter(extractor.graph_layout, epsilon=t, alpha=0.4), BoxPainter(), CustomTextPainter((50, 50), 'rotations', child_only=True)]
    global_painters = [GlobalPainter(p) for p in local_painters]
    graphs = [DynamicPolar('AUs', child_aus, data['au_cols'], height=height // 2, width=width // 2, filters=()),
              DynamicSignal('Emotions', child_emotions, data['emotion_cols'],
                            'Time', 'Score',
                            window_size=1000,
                            height=height // 2,
                            width=width // 2,
                            filters=())]
    vc = VideoCreator(global_painters=global_painters, local_painters=[], graphs=graphs, scale=scale)
    vc.create_video(video_path=video_path, data=data, out_path=out_path, start=start, end=end, unit='frame')


def create_jordi(cfg, out_path, start=None, end=None, scale=1):
    extractor = MMPoseDataExtractor(COCO_LAYOUT, scale=scale)
    data = extractor(skeleton_path, predictions_path)
    width, height = data['resolution']
    epsilon = 0.3
    # local_painters = [GraphPainter(extractor.graph_layout, epsilon=epsilon, alpha=0.4), LabelPainter(), ScorePainter(), BoxPainter()]
    # local_painters = [GraphPainter(extractor.graph_layout, epsilon=epsilon, alpha=1.0, color=[(255, 128, 0), (0, 255, 128), (128, 0, 255)], child_only=False)]
    local_painters = [GraphPainter(extractor.graph_layout, epsilon=cfg.painters['local']['GraphPainter']['epsilon'], alpha=cfg.painters['local']['GraphPainter']['alpha'], child_only=cfg.painters['local']['GraphPainter']['child_only'])]
    global_painters = [BlurPainter(data), ScaleAbsPainter(alpha=cfg.painters['global']['ScaleAbsPainter']['alpha'], beta=cfg.painters['global']['ScaleAbsPainter']['alpha'])] + [GlobalPainter(p) for p in local_painters]
    graphs = [DynamicSignal(None, data['predictions'], None,
                            'Time', 'Score',
                            window_size=1000,
                            height=int(height * 0.5),
                            width=width,
                            filters=(),
                            xlabel_format='time'),
              DynamicSkeleton(layout=extractor.graph_layout, epsilon=epsilon, data=data, child_only=True,
                              height=int(height * 0.5), width=width)]

    vc = VideoCreator(global_painters=global_painters, local_painters=[], graphs=graphs, scale=scale)
    vc.create_video(video_path=video_path, data=data, out_path=out_path, start=start, end=end, unit='frame')

def create_jordi_ann(cfg):
    video_path, processed_dir, out_dir = cfg.paths.video, cfg.paths.processed, cfg.paths.output
    name, ext = osp.splitext(osp.basename(video_path))
    ann_file = osp.join(processed_dir, name, 'jordi', 'cv0.pth', f'{name}_annotations.csv')
    if not osp.exists(ann_file):
        print(f'Annotations for {name} do not exist.')
    video_info = load_config(osp.join(processed_dir, name, 'jordi', 'cv0.pth', f'{name}_exec_info.yaml'))
    df = pd.read_csv(ann_file)
    df = df[df['movement'] == 'Stereotypical']
    out_dir = osp.join(out_dir, f'{name}_Stereotypical')
    init_directories(out_dir)
    for i, row in tqdm(df.iterrows()):
        start, end = max(0, int(row['start_frame']) - 200), min(video_info['properties']['frame_count'], int(row['end_frame']) + 200)
        out_path = osp.join(out_dir, f'{name}_{start}_{end}{ext}')
        create_jordi(cfg, out_path, start=start, end=end)


if __name__ == '__main__':
    logger = init_logger('Visualizer')
    parser = ArgumentParser()
    parser.add_argument('-cfg', '--config_path')
    args = parser.parse_args()
    cfg = load_config(args.config_path)
    cfg = OmegaConf.to_container(cfg)
    logger.info(f'Visualizer initialized with config: {cfg}')

    loader = ComponentsLoader(cfg)
    data = loader.extract_data()
    local_painters, global_painters = loader.create_painters(use_globals=cfg['painters']['use_globals'])
    graphs = loader.create_graphs()
    vc = VideoCreator(global_painters=global_painters, local_painters=local_painters, graphs=graphs, scale=1)

    video_path, processed_dir, out_dir = cfg['paths']['video'], cfg['paths']['processed'], cfg['paths']['output']
    name, ext = cfg['name'], cfg['ext']
    model = cfg['model']['name']
    mode = cfg['mode']
    if mode['name'] == 'annotate':
        ann_file = osp.join(processed_dir, name, 'jordi', 'cv0.pth', f'{name}_annotations.csv')
        if not osp.exists(ann_file):
            logger.error(f'Annotations for {cfg["name"]} do not exist.')
        else:
            video_info = load_config(osp.join(processed_dir, name, 'jordi', 'cv0.pth', f'{name}_exec_info.yaml'))
            df = pd.read_csv(ann_file)
            df = df[df['movement'] == 'Stereotypical'].reset_index(drop=True)
            out_dir = osp.join(cfg['paths']['output'], f'{name}_{model}')
            init_directories(out_dir)
            for i, row in tqdm(df.iterrows()):
                start, end = max(0, int(row['start_frame']) - 200), min(video_info['properties']['frame_count'], int(row['end_frame']) + 200)
                out_path = osp.join(out_dir, f'{name}{ext}')
                vc.create_video(video_path=cfg['paths']['video'], data=data, out_path=out_path, start=start, end=end, unit='frame')
    elif mode['name'] == 'video':
        start, end = (mode['start'], mode['end']) if 'start' in mode and 'end' in mode else (None, None)
        out_path = osp.join(out_dir, f'{name}_{model}{ext}')
        vc.create_video(video_path=cfg['paths']['video'], data=data, out_path=out_path, start=start, end=end, unit='frame')

# if __name__ == '__main__':
#     logger = init_logger('Visualizer')
#     parser = ArgumentParser()
#     parser.add_argument('-cfg', '--config_path')
#     # parser.add_argument("-v", "--video_path")
#     # parser.add_argument("-p", "--processed_dir", default=r'Z:\Users\TalBarami\models_outputs\processed')
#     # parser.add_argument("-out", "--out_dir")
#     # parser.add_argument("-s", "--start")
#     # parser.add_argument("-t", "--end")
#     # parser.add_argument("-a", "--ann", action='store_true')
#     # parser.add_argument('-j', '--jordi', action='store_true')
#     # parser.add_argument('-b', '--barni', action='store_true')
#     # parser.add_argument('-c', '--scale', type=int, default=1)
#     args = parser.parse_args()
#
#     cfg = load_config(args.config_path)
#
#     logger.info(f'Visualizer initialized with config:{cfg}')
#     name, ext = osp.splitext(osp.basename(cfg.paths.video))
#     if cfg.model.method == 'annotations':
#         create_jordi_ann(cfg)
#     else:
#         if args.jordi:
#             out_path = osp.join(args.out_dir, f'{name}_SMMs.avi')
#             logger.info(f'Creating Jordi video for {name}')
#             create_jordi(video_path=args.video_path, processed_dir=args.processed_dir, out_path=out_path,
#                          start=args.start, end=args.end, scale=args.scale)
#         if args.barni:
#             out_path = osp.join(args.out_dir, f'{name}_Facial.mp4')
#             logger.info(f'Creating Barni video for {name}')
#             create_barni(video_path=args.video_path, processed_dir=args.processed_dir, out_path=out_path,
#                          start=args.start, end=args.end, scale=args.scale)