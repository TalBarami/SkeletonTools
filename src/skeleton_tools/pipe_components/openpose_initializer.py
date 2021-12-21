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

from skeleton_tools.pipe_components.yolo_v5_child_detector import ChildDetector
from skeleton_tools.utils.constants import LENGTH, JSON_SOURCES, EPSILON
from skeleton_tools.utils.skeleton_utils import normalize_json
from skeleton_tools.utils.tools import read_json, write_json, get_video_properties, write_pkl


class SkeletonSource(Enum):
    VIDEO = 'video'
    IMAGE = 'image_dir'
    WEBCAM = 'camera'


class OpenposeInitializer:
    def __init__(self, openpose_layout, in_channels=3, length=LENGTH, num_person_in=5, num_person_out=1, open_pose_path='C:/research/openpose'):
        self.layout = openpose_layout
        self.C = in_channels
        self.T = length
        self.V = len(self.layout)
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
        self.open_pose_path = open_pose_path

    def _exec_openpose(self, src_path, skeleton_dst=None, source_type=SkeletonSource.VIDEO):
        Path(skeleton_dst).mkdir(parents=True, exist_ok=True)
        params = {
            source_type.value: f'\"{src_path}\"',
            'model_pose': self.layout.model_pose,
            'write_json': f'\"{skeleton_dst}\"',
            'display': 0,
            'render_pose': 0
        }

        if self.layout.face:
            params['face'] = ''
            params['net_resolution'] = '256x256'
        if self.layout.hand:
            params['hand'] = ''
        if self.layout.name == 'BODY_21A':
            params['tracking'] = 1

        args = ' '.join([f'--{k} {v}' for k, v in params.items()])

        cwd = os.getcwd()
        os.chdir(self.open_pose_path)
        cmd = f'build_windows/x64/Release/OpenPoseDemo.exe {args}' if path.exists('build_windows') else f'bin/OpenPoseDemo.exe {args}'
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
            for pdx, p in enumerate(people):
                skeleton = {'person_id': p['person_id'] if p['person_id'] != [-1] else pdx}
                for source in [s for s in JSON_SOURCES if s['openpose'] in p.keys()]:
                    pose, score = collect_data(p[source['openpose']])
                    skeleton[source['name']] = pose
                    skeleton[f'{source["name"]}_score'] = score
                skeletons.append(skeleton)
            result.append({'frame_index': i,
                           'skeleton': skeletons})
        return result

    # def openpose_to_numpy(self, openpose_dir):
    #     file_names = [path.join(openpose_dir, f) for f in os.listdir(openpose_dir) if path.isfile(path.join(openpose_dir, f)) and f.endswith('json')]
    #
    #     def append_info(pose, score, p, key):
    #         pose = pose + list(zip(np.round(p[key][::3], 8), np.round(p[key][1::3], 8)))
    #         score = np.concatenate((score, np.round(p[key][2::3], 8)))
    #         return np.array(pose), score
    #
    #     T = len(file_names)
    #     kp = np.zeros((T, self.num_person_in, self.V, self.C - 1))
    #     scores = np.zeros((T, self.num_person_in, self.V))
    #     pids = np.zeros((T, self.num_person_in))
    #
    #     for i, file in tqdm(enumerate(file_names), ascii=True, desc='Openpose to numpy'):
    #         frame_info = read_json(file)
    #         people = frame_info['people']
    #
    #         # frame_kp = []
    #         # frame_scores = []
    #         # frame_pids = []
    #         for pidx, p in enumerate(people):
    #             if pidx > self.num_person_in:
    #                 raise IndexError("Reached maximum number of people")
    #             pose, score = append_info([], np.array([]), p, 'pose_keypoints_2d')
    #             # pose = list(zip(np.round(p['pose_keypoints_2d'][::3], 8), np.round(p['pose_keypoints_2d'][1::3], 8)))
    #             # score = np.round(p['pose_keypoints_2d'][2::3], 8)
    #             if self.layout.face:
    #                 pose, score = append_info(pose, score, p, 'face_keypoints_2d')
    #                 # pose += list(zip(np.round(p['face_keypoints_2d'][::3], 8), np.round(p['face_keypoints_2d'][1::3], 8)))
    #                 # score = np.concatenate((score, np.round(p['face_keypoints_2d'][2::3], 8)))
    #             if self.layout.hand:
    #                 pose, score = append_info(pose, score, p, 'hand_left_keypoints_2d')
    #                 pose, score = append_info(pose, score, p, 'hand_right_keypoints_2d')
    #                 # pose += list(zip(np.round(p['hand_left_keypoints_2d'][::3], 8), np.round(p['hand_left_keypoints_2d'][1::3], 8)))
    #                 # score = np.concatenate((score, np.round(p['hand_left_keypoints_2d'][2::3], 8)))
    #                 # pose += list(zip(np.round(p['hand_right_keypoints_2d'][::3], 8), np.round(p['hand_right_keypoints_2d'][1::3], 8)))
    #                 # score = np.concatenate((score, np.round(p['hand_right_keypoints_2d'][2::3], 8)))
    #             kp[i, pidx] = pose
    #             scores[i, pidx] = score
    #             pids[i, pidx] = p['person_id'] if p['person_id'] != [-1] else pidx
    #     return kp, scores, pids

    # def collect_skeleton(self, skeleton_folder, dst_folder, resolution, tracking=True):  # TODO: Remove tracking, remove labeling
    #     skeleton = self.openpose_to_json(skeleton_folder)
    # skeleton = self.normalize_pose(skeleton, resolution)
    # if tracking:
    #     skeleton = track(skeleton, skeleton_folder, resolution, self.layout)
    # result = {'data': self.normalize_pose(skeleton, resolution),
    #           'label': label_name,
    #           'label_index': int(label_index)}
    # tools.write_json(skeleton, path.join(dst_folder, f'{path.basename(skeleton_folder)}.json'))

    def prepare_skeleton(self, src_path, result_skeleton_dir=None, source_type=SkeletonSource.VIDEO, out_name=None):
        basename = path.basename(src_path)
        basename_no_ext = path.splitext(basename)[0] if source_type == SkeletonSource.VIDEO else basename
        openpose_output_path = path.join(self.open_pose_path, 'runs', basename_no_ext) if result_skeleton_dir is None else path.join(result_skeleton_dir, 'openpose', basename_no_ext)

        try:
            resolution, fps, frame_count = get_video_properties(src_path)
            self._exec_openpose(src_path, openpose_output_path, source_type=source_type)
            # kp, scores, pids = self.openpose_to_numpy(openpose_output_path)
            data = self.openpose_to_json(openpose_output_path)
            skeleton = {
                'name': basename,
                'resolution': resolution,
                'fps': fps,
                'length': frame_count,
                'data': data
            }
            if result_skeleton_dir:
                result_path = path.join(result_skeleton_dir, out_name if out_name else f'{basename_no_ext}.json')
                write_pkl(skeleton, result_path)
            return skeleton
        except Exception as e:
            print(f'Error creating skeleton from {src_path}: {e}')
            raise e
        finally:
            shutil.rmtree(openpose_output_path)

    def normalize(self, pose_json, resolution):
        return normalize_json(pose_json, resolution)

    def denormalize(self, data_numpy, resolution):
        x = data_numpy.copy()
        width, height = resolution
        x[:2] += 0.5
        x[0][x[2] < EPSILON] = 0
        x[1][x[2] < EPSILON] = 0
        x[0] *= width
        x[1] *= height
        return x

    def to_numpy(self, skeleton):
        data_numpy = np.zeros((self.C, self.T, self.V, self.num_person_in))
        for i, frame_info in tqdm(enumerate(skeleton), ascii=True, desc='To numpy'):
            if i == self.T:
                break
            for m, skeleton_info in enumerate(frame_info["skeleton"]):  # TODO: case when id > num_person_in, but len(skeleton) < num_person_in
                pid = m if ('person_id' not in skeleton_info.keys()) else skeleton_info['person_id']
                if type(pid) == list:
                    pid = m if pid[0] < 0 else pid[0]
                pid %= self.num_person_in
                pose = skeleton_info['pose']
                score = skeleton_info['score'] if 'score' in skeleton_info.keys() else skeleton_info['pose_score']
                data_numpy[0, i, :, pid] = pose[0::2]
                data_numpy[1, i, :, pid] = pose[1::2]
                data_numpy[2, i, :, pid] = score

        sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
        for t, s in enumerate(sort_index):
            data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2, 0))
        return data_numpy[:, :, :, 0:self.num_person_out]
