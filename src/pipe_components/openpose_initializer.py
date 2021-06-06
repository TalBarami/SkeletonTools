import os
import shlex
import shutil
import subprocess
from enum import Enum
from itertools import chain
from os import path
from pathlib import Path
import numpy as np
from copy import deepcopy
from PIL import Image

from utils import tools
from utils.constants import JSON_SOURCES
from utils.tools import read_json


class SkeletonSource(Enum):
    VIDEO = 'video'
    IMAGE = 'image_dir'
    WEBCAM = 'camera'

class VideoPreparator:
    def __init__(self, openpose_layout):
        self.layout = openpose_layout

    def make_skeleton(self, src_path, skeleton_dst, source_type=SkeletonSource.VIDEO, render_pose=False, face=False, hand=False, open_pose_path='C:/research/openpose', write_video=None):
        Path(skeleton_dst).mkdir(parents=True, exist_ok=True)
        params = {
            source_type: f'\"{src_path}\"',
            'model_pose': self.layout.name,
            'write_json': f'\"{skeleton_dst}\"',
            'display': 0,
            'render_pose': render_pose
        }
        if write_video:
            params['write_video'] = f'\"{write_video}\"'
            if source_type == SkeletonSource.IMAGE:
                params['write_video_fps'] = 30.0
        if face:
            params['face'] = ''
            params['net_resolution'] = '256x256'
        if hand:
            params['hand'] = ''
        if self.layout.name == 'BODY_21A':
            params['tracking'] = 1

        args = ' '.join([f'--{k} {v}' for k, v in params.items()])

        cwd = os.getcwd()
        os.chdir(open_pose_path)
        cmd = f'build_windows/x64/Release/OpenPoseDemo.exe {args}'
        print(f'Executing: {cmd}')
        try:
            subprocess.check_call(shlex.split(cmd), universal_newlines=True)
        finally:
            os.chdir(cwd)

    def openpose_to_json(self, openpose_dir):
        file_names = [path.join(openpose_dir, f) for f in os.listdir(openpose_dir) if path.isfile(path.join(openpose_dir, f)) and f.endswith('json')]

        def collect_data(lst):
            k = np.array([float(c) for c in lst])
            x = np.round(k[::3], 8)
            y = np.round(k[1::3], 8)
            c = np.round(k[2::3], 8).tolist()
            return list(chain(*[(_x, _y) for (_x, _y) in zip(x, y)])), c

        result = []
        for i, file in enumerate(file_names):
            skeletons = []
            frame_info = read_json(file)
            people = frame_info['people']
            for p in people:
                skeleton = {'person_id': p['person_id']}
                for source in [s for s in JSON_SOURCES if s['openpose'] in p.keys()]:
                    pose, score = collect_data(p[source['openpose']])
                    skeleton[source['name']] = pose
                    skeleton[f'{source["name"]}_score'] = score
                skeletons.append(skeleton)
            result.append({'frame_index': i,
                           'skeleton': skeletons})
        return result

    def collect_skeleton(self, skeleton_folder, dst_folder, label_index, label_name, resolution, tracking=True):  # TODO: Remove tracking, remove labeling
        skeleton = self.openpose_to_json(skeleton_folder)
        if tracking:
            skeleton = track(skeleton, skeleton_folder, resolution, self.layout)
        result = {'data': self.normalize_pose(skeleton, resolution),
                  'label': label_name,
                  'label_index': int(label_index)}
        tools.write_json(result, path.join(dst_folder, f'{path.basename(skeleton_folder)}.json'))

    def normalize_pose(self, pose_json, resolution):
        result = deepcopy(pose_json)
        width, height = resolution
        for d in result:
            for s in d['skeleton']:
                for src in [src for src in JSON_SOURCES if src['name'] in s.keys()]:
                    x = np.round(np.array(s[src['name']][::2]) / width, 8)
                    y = np.round(np.array(s[src['name']][1::2]) / height, 8)
                    s[src['name']] = [e for l in zip(x, y) for e in l]
        return result

    def prepare_skeleton(self, src_path, result_skeleton_dir, label_index, label_name, source_type=SkeletonSource.VIDEO, resolution=None, result_video_path=None, face=False, hand=False, tracking=True):
        basename = path.basename(src_path)
        basename_no_ext = path.splitext(basename)[0] if source_type == SkeletonSource.VIDEO else basename

        openpose_output_path = path.join(result_skeleton_dir, 'openpose', basename_no_ext)

        if result_video_path is not None:
            Path(path.join(result_video_path)).mkdir(parents=True, exist_ok=True)
            result_video_path = path.join(result_video_path, f'{basename_no_ext}.avi')

        try:
            if resolution is None:
                if source_type == SkeletonSource.VIDEO:
                    resolution, _, _ = tools.get_video_properties(src_path)
                else:
                    sample_image = path.join(src_path, os.listdir(src_path)[0])
                    with Image.open(sample_image) as img:
                        resolution = img.size
            self.make_skeleton(src_path, openpose_output_path, source_type=source_type, write_video=result_video_path, face=face, hand=hand)
            self.collect_skeleton(openpose_output_path, result_skeleton_dir, label_index, label_name, resolution, tracking=tracking)
            shutil.rmtree(openpose_output_path)
        except Exception as e:
            print(f'Error creating skeleton from {src_path}: {e}')
            raise e
        # finally:
        #     shutil.rmtree(openpose_output_path)
