import os
import shlex
import shutil
import subprocess
from copy import deepcopy
from os import path

import numpy as np

from skeleton_tools.utils.skeleton_utils import bounding_box, box_distance, normalize_json
from skeleton_tools.utils.tools import read_json, get_video_properties


class ChildDetector:
    def __init__(self,
                 detector_root=r"C:\research\yolov5",
                 model_path=r"C:\research\yolov5\runs\train\exp7\weights\best.pt",
                 ffmpeg_root=r"C:\research\ffmpeg-N-101443-g74b5564fb5-win64-gpl\bin",
                 resolution=(1280, 1024),
                 data_centralized=False):
        self.detector_root = detector_root
        self.detection_dir = path.join(self.detector_root, 'runs', 'detect')
        self.temp_dir = path.join(self.detector_root, 'runs', 'temp')
        self.model_path = model_path
        self.ffmpeg_root = ffmpeg_root
        self.resolution = resolution
        self.data_centralized = data_centralized

    def _rescale_video(self, video_path, out_path):
        width, height = self.resolution
        subprocess.check_call(' '.join([path.join(self.ffmpeg_root, 'ffmpeg.exe'), '-i', f'"{video_path}"', '-vf', f'scale={width}:{height}', f'"{out_path}"']))

    def _read_box(self, box_path):
        with open(box_path, 'r') as f:
            # children = [x.strip() for x in f.readlines() if x[0] == '1']
            c_boxes = [[float(s) for s in x.strip().split(' ')[1:]] for x in f.readlines() if x[0] == '1']
        return [(np.array((cx, cy)), np.array((w, h))) for cx, cy, w, h in c_boxes]

    def _choose_box(self, boxes, prev_box=None):
        new_box = []
        if len(boxes) == 0 and prev_box is not None:
            return prev_box
        elif len(boxes) > 0:
            if len(boxes) > 1 and prev_box is not None:
                distances = [box_distance(b, prev_box) for b in boxes]
                new_box = boxes[np.argmin(distances)]
            else:
                new_box = boxes[0]
        return new_box

    def _collect_json(self, label_path):
        data = []
        box = None
        last_known_box = box
        for frame_index, file in enumerate(os.listdir(label_path)):
            children = self._read_box(path.join(label_path, file))
            if box is not None and len(box) > 0:
                last_known_box = box
            box = self._choose_box(children, last_known_box)
            if self.data_centralized:
                box[:2] -= 0.5
            data.append({
                'frame_index': frame_index,
                'box': box
            })
        return data

    def _detect_children_in_video(self, video_path, resolution=None):
        name, ext = path.splitext(path.basename(video_path))
        temp_scaled = path.join(self.temp_dir, f'{name}{ext}')
        width, height = self.resolution if resolution is None else resolution
        if path.exists(temp_scaled):
            os.remove(temp_scaled)
        if path.exists(path.join(self.detection_dir, name)):
            shutil.rmtree(path.join(self.detection_dir, name))

        try:
            vid_res, _, _ = get_video_properties(video_path)
            if vid_res != self.resolution:
                raise ValueError("Resolution mismatch")
            # self._rescale_video(video_path, temp_scaled)
            args = {
                'weights': f'"{self.model_path}"',
                'img': width,
                'source': f'"{video_path}"',
                'save-txt': '',
                'nosave': '',
                'project': f'"{self.detection_dir}"',
                'name': f'"{name}"'
            }
            python_path = r'C:\Users\owner\anaconda3\envs\yolo\python.exe'
            cmd = f'"{python_path}" "{path.join(self.detector_root, "detect.py")}" {" ".join([f"--{k} {v}" for k, v in args.items()])}'
            print(f'Executing: {cmd}')
            subprocess.check_call(shlex.split(cmd), universal_newlines=True)
            # subprocess.check_call(cmd)
            box_json = self._collect_json(path.join(self.detection_dir, name, 'labels'))
            return box_json
        finally:
            if path.exists(temp_scaled):
                os.remove(temp_scaled)
            if path.exists(path.join(self.detection_dir, name)):
                shutil.rmtree(path.join(self.detection_dir, name))

    def _match_skeleton(self, box, skeletons):
        if box['box']:
            (cx, cy), (w, h) = box['box']
            distances = [box_distance((np.array([cx, cy]), np.array([w, h])), bounding_box((np.array([skel['pose'][0::2], skel['pose'][1::2]]).T / np.array(self.resolution)).T,
                                                                                           np.array(skel['pose_score']))) for skel in skeletons]
            return skeletons[np.argmin(distances)]['person_id']

    def _match_video(self, box_json, video_json):
        if (not (any(box_json) or any(video_json))) or box_json[-1]['frame_index'] != video_json[-1]['frame_index']:
            raise ValueError(f'Box: {len(box_json)}, Skeleton: {len(video_json)}')
        pids = [self._match_skeleton(box, frame_info['skeleton']) if frame_info['skeleton'] else -1
                for box, frame_info in zip(box_json, video_json)]
        return pids

    def remove_adults(self, skeleton, video_path, resolution=None):
        box_json = self._detect_children_in_video(video_path, resolution=resolution)
        cids = self._match_video(box_json, skeleton)
        data = [{'frame_index': frame_info['frame_index'], 'skeleton': [s for s in frame_info['skeleton'] if s['person_id'] == cid]}
                for frame_info, cid in zip(skeleton, cids)]
        return data
