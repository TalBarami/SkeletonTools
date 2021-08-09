import os
import shlex
import shutil
import subprocess
from os import path

import numpy as np

from skeleton_tools.utils.skeleton_utils import bounding_box, box_distance
from skeleton_tools.utils.tools import read_json


class ChildDetector:
    def __init__(self,
                 detector_root=r"C:\research\yolov5",
                 model_path=r"C:\research\yolov5\runs\train\exp7\weights\best.pt",
                 ffmpeg_root=r"C:\research\ffmpeg-N-101443-g74b5564fb5-win64-gpl\bin",
                 resolution=(1280, 1024)):
        self.detector_root = detector_root
        self.detection_dir = path.join(self.detector_root, 'runs', 'detect')
        self.temp_dir = path.join(self.detector_root, 'runs', 'temp')
        self.model_path = model_path
        self.ffmpeg_root = ffmpeg_root
        self.resolution = resolution

    def _rescale_video(self, video_path, out_path):
        width, height = self.resolution
        subprocess.check_call(' '.join([path.join(self.ffmpeg_root, 'ffmpeg.exe'), '-i', f'"{video_path}"', '-vf', f'scale={width}:{height}', f'"{out_path}"']))

    def _read_box(self, box_path):
        with open(box_path, 'r') as f:
            children = [x.strip() for x in f.readlines() if x[0] == '1']
        return children

    def _collect_json(self, label_path):
        data = []
        for frame_index, file in enumerate(os.listdir(label_path)):
            children = self._read_box(path.join(label_path, file))
            box = [float(x) for x in children[0][2:].split(' ')] if len(children) > 0 else []
            if len(children) > 1:
                print(f'Multiple children at frame {frame_index}: {label_path}')
            data.append({
                'frame_index': frame_index,
                'box': box
            })
        return data

    def _detect_children_in_video(self, video_path):
        name, ext = path.splitext(path.basename(video_path))
        temp_scaled = path.join(self.temp_dir, f'{name}{ext}')
        width, height = self.resolution
        if path.exists(temp_scaled):
            os.remove(temp_scaled)
        if path.exists(path.join(self.detection_dir, name)):
            shutil.rmtree(path.join(self.detection_dir, name))

        try:
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
            cmd = f'python {path.join(self.detector_root, "detect.py")} {" ".join([f"--{k} {v}" for k, v in args.items()])}'
            print(f'Executing: {cmd}')
            subprocess.check_call(cmd)
            box_json = self._collect_json(path.join(self.detection_dir, name, 'labels'))
            return box_json
        finally:
            if path.exists(temp_scaled):
                os.remove(temp_scaled)
            if path.exists(path.join(self.detection_dir, name)):
                shutil.rmtree(path.join(self.detection_dir, name))

    def _match_skeleton(self, box, skeletons):
        if box:
            cx, cy, w, h = box['box']
            distances = [box_distance((np.array([cx, cy]), np.array([w, h])), bounding_box(np.array([skel['pose'][0::2], skel['pose'][1::2]]), np.array(skel['score']))) for skel in skeletons]
            return skeletons[np.argmin(distances)]['person_id']

    def _match_video(self, box_json, video_json):
        if box_json[-1]['frame_index'] != video_json[-1]['frame_index']:
            print('Frame count mismatch')
        pids = [self._match_skeleton(box, frame_info['skeleton']) if frame_info['skeleton'] else -1
                for box, frame_info in zip(box_json, video_json)]
        return pids

    def detect_pids(self, skeleton_json_path, video_path):
        box_json = self._detect_children_in_video(video_path)
        cids = self._match_video(box_json, read_json(skeleton_json_path)['data'])
        return cids


if __name__ == '__main__':
    d = ChildDetector()
    name = '22210917_ADOS_Clinical_120320_0938_1_Hand flapping_3847_4029'
    cids = d.detect_pids(path.join(r'D:\datasets\autism_center\skeletons\data', f'{name}.json'),
                         path.join(r'D:\datasets\autism_center\segmented_videos', f'{name}.avi'))
    print(1)