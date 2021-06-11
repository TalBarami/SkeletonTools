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
from tqdm import tqdm

from utils import tools
from utils.constants import JSON_SOURCES, EPSILON, LENGTH
from utils.tools import read_json


class SkeletonSource(Enum):
    VIDEO = 'video'
    IMAGE = 'image_dir'
    WEBCAM = 'camera'

class OpenposeInitializer:
    def __init__(self, openpose_layout):
        self.layout = openpose_layout
        self.C = 3
        self.T = LENGTH
        self.V = len(self.layout)
        self.num_person_in = 5
        self.num_person_out = 2

    def make_skeleton(self, src_path, skeleton_dst, source_type=SkeletonSource.VIDEO, render_pose=False, face=False, hand=False, open_pose_path='C:/research/openpose', write_video=None):
        Path(skeleton_dst).mkdir(parents=True, exist_ok=True)
        params = {
            source_type.value: f'\"{src_path}\"',
            'model_pose': self.layout.name,
            'write_json': f'\"{skeleton_dst}\"',
            'display': 0,
            'render_pose': int(render_pose)
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
            print('OpenPose finished.')

    def openpose_to_json(self, openpose_dir):
        file_names = [path.join(openpose_dir, f) for f in os.listdir(openpose_dir) if path.isfile(path.join(openpose_dir, f)) and f.endswith('json')]

        def collect_data(lst):
            k = np.array([float(c) for c in lst])
            x = np.round(k[::3], 8)
            y = np.round(k[1::3], 8)
            c = np.round(k[2::3], 8).tolist()
            return list(chain(*[(_x, _y) for (_x, _y) in zip(x, y)])), c

        result = []
        for i, file in tqdm(enumerate(file_names), ascii=True, desc='Openpose to JSON'):
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

    # def collect_skeleton(self, skeleton_folder, dst_folder, resolution, tracking=True):  # TODO: Remove tracking, remove labeling
    #     skeleton = self.openpose_to_json(skeleton_folder)
        # skeleton = self.normalize_pose(skeleton, resolution)
        # if tracking:
        #     skeleton = track(skeleton, skeleton_folder, resolution, self.layout)
        # result = {'data': self.normalize_pose(skeleton, resolution),
        #           'label': label_name,
        #           'label_index': int(label_index)}
        # tools.write_json(skeleton, path.join(dst_folder, f'{path.basename(skeleton_folder)}.json'))

    def prepare_skeleton(self, src_path, result_skeleton_dir, source_type=SkeletonSource.VIDEO, resolution=None, result_video_path=None, face=False, hand=False, tracking=True):
        basename = path.basename(src_path)
        basename_no_ext = path.splitext(basename)[0] if source_type == SkeletonSource.VIDEO else basename

        openpose_output_path = path.join(result_skeleton_dir, 'openpose', basename_no_ext)

        # if result_video_path is not None:
        #     Path(path.join(result_video_path)).mkdir(parents=True, exist_ok=True)
        #     result_video_path = path.join(result_video_path, f'{basename_no_ext}.avi')

        try:
            # if resolution is None:
            #     if source_type == SkeletonSource.VIDEO:
            #         resolution, _, _ = tools.get_video_properties(src_path)
            #     else:
            #         sample_image = path.join(src_path, os.listdir(src_path)[0])
            #         with Image.open(sample_image) as img:
            #             resolution = img.size
            self.make_skeleton(src_path, openpose_output_path, source_type=source_type, write_video=result_video_path, face=face, hand=hand)
            tools.write_json(self.openpose_to_json(openpose_output_path), path.join(result_skeleton_dir, f'{basename_no_ext}.json'))
            # shutil.rmtree(openpose_output_path)
        except Exception as e:
            print(f'Error creating skeleton from {src_path}: {e}')
            raise e
        # finally:
        #     shutil.rmtree(openpose_output_path)

    def normalize(self, pose_json, resolution):
        result = deepcopy(pose_json)
        width, height = resolution
        for d in tqdm(result, ascii=True, desc='Normalizing & Centralizing'):
            for s in d['skeleton']:
                for src in [src for src in JSON_SOURCES if src['name'] in s.keys()]:
                    key = src['name']
                    c = np.array(s[f'{key}_score'])
                    xy = np.array([s[key][::2], s[key][1::2]])
                    xy = np.round((xy.T / np.array([width, height])).T - 0.5, 8)
                    xy.T[c < EPSILON] = 0
                    x, y = xy[0], xy[1]
                    s[src['name']] = [e for l in zip(x, y) for e in l]
        return result

    def to_numpy(self, skeleton):
        data_numpy = np.zeros((self.C, self.T, self.V, self.num_person_in))
        for frame_info in tqdm(skeleton, ascii=True, desc='To numpy'):
            frame_index = int(frame_info['frame_index'])
            if frame_index == self.T:
                break
            for m, skeleton_info in enumerate(frame_info["skeleton"]):  # TODO: case when id > num_person_in, but len(skeleton) < num_person_in
                pid = m if ('person_id' not in skeleton_info.keys()) else skeleton_info['person_id']
                if type(pid) == list:
                    pid = m if pid[0] < 0 else pid[0]
                pid %= self.num_person_in
                pose = skeleton_info['pose']
                score = skeleton_info['score'] if 'score' in skeleton_info.keys() else skeleton_info['pose_score']
                data_numpy[0, frame_index, :, pid] = pose[0::2]
                data_numpy[1, frame_index, :, pid] = pose[1::2]
                data_numpy[2, frame_index, :, pid] = score

        sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
        for t, s in enumerate(sort_index):
            data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2, 0))
        return data_numpy[:, :, :, 0:self.num_person_out]
