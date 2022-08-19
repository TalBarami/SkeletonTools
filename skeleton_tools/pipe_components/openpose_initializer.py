import os
import shlex
import shutil
import subprocess
from enum import Enum
from itertools import chain
from os import path as osp
from pathlib import Path

import cv2
import numpy as np
from copy import deepcopy
from PIL import Image
from tqdm import tqdm

from skeleton_tools.openpose_layouts.body import BODY_25_LAYOUT, COCO_LAYOUT
from skeleton_tools.openpose_layouts.graph_layout import convert_layout
from skeleton_tools.utils.constants import LENGTH, JSON_SOURCES, EPSILON, OPENPOSE_ROOT
from skeleton_tools.utils.skeleton_utils import normalize_json
from skeleton_tools.utils.tools import read_json, write_json, get_video_properties, write_pkl, init_directories, init_logger


class SkeletonSource(Enum):
    VIDEO = 'video'
    IMAGE = 'image_dir'
    WEBCAM = 'camera'


class OpenposeInitializer:
    def __init__(self, openpose_layout, in_channels=3, length=LENGTH, num_person_in=5, num_person_out=5, open_pose_path=OPENPOSE_ROOT, as_img_dir=False, logger=None):
        if logger is None:
            self.logger = init_logger('OpenPoseInitializer')
        else:
            self.logger = logger
        self.layout = openpose_layout
        self.C = in_channels
        self.T = length
        self.V = len(self.layout)
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
        self.open_pose_path = open_pose_path
        self.as_img_dir = as_img_dir

    def _video2img(self, video_path, out_path):
        name = osp.splitext(osp.basename(video_path))[0]
        self.logger.info(f'Converting video to image dir.')
        init_directories(out_path)
        cap = cv2.VideoCapture(video_path)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        d = len(str(n))
        i, ret = 0, True
        while ret:
            ret, frame = cap.read()
            cv2.imwrite(osp.join(out_path, f'{name}_{str(i).zfill(d)}.jpg'), frame)
            i += 1

    def _exec_openpose(self, src_path, skeleton_dst, source_type=SkeletonSource.VIDEO):
        init_directories(skeleton_dst)
        if src_path.startswith('\\\\'):
            src_path = f'\\{src_path}'
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
        cmd = f'build_windows/x64/Release/OpenPoseDemo.exe {args}' if osp.exists('build_windows') else f'bin/OpenPoseDemo.exe {args}'
        self.logger.info(f'Executing: {cmd}')
        try:
            subprocess.check_call(shlex.split(cmd), universal_newlines=True)
        finally:
            os.chdir(cwd)
            self.logger.info('OpenPose finished.')

    def prepare_skeleton(self, src_path, result_skeleton_dir=None, source_type=SkeletonSource.VIDEO, out_name=None):
        basename = osp.basename(src_path)
        basename_no_ext = osp.splitext(basename)[0] if source_type == SkeletonSource.VIDEO else basename

        process_dir = osp.join(self.open_pose_path, 'runs', basename_no_ext) if result_skeleton_dir is None else osp.join(result_skeleton_dir, basename_no_ext)
        openpose_output_path = osp.join(process_dir, 'openpose')

        try:
            resolution, fps, frame_count, length = get_video_properties(src_path, method='cv2')
            if self.as_img_dir:
                img_out_path = osp.join(process_dir, 'img_dirs')
                self._video2img(src_path, img_out_path)
                self._exec_openpose(img_out_path, openpose_output_path, source_type=SkeletonSource.IMAGE)
            else:
                self._exec_openpose(src_path, openpose_output_path, source_type=source_type)
            data = self.openpose_to_json(openpose_output_path)
            adj = int(frame_count - len(data))
            if adj != 0:
                self.logger.warning(f'Skeleton {basename} requires adjustments.')
            skeleton = {
                'name': basename,
                'video_path': src_path,
                'resolution': resolution,
                'fps': fps,
                'frame_count': frame_count,
                'length_seconds': length,
                'adjust': adj,
                'data': data,
            }
            if result_skeleton_dir:
                result_path = osp.join(result_skeleton_dir, out_name if out_name else f'{basename_no_ext}.json')
                write_pkl(skeleton, result_path)
            return skeleton
        except Exception as e:
            self.logger.error(f'Error creating skeleton from {src_path}: {e}')
            raise e
        finally:
            shutil.rmtree(process_dir)

    def openpose_to_json(self, openpose_dir):
        file_names = [path.join(openpose_dir, f) for f in os.listdir(openpose_dir) if osp.isfile(path.join(openpose_dir, f)) and f.endswith('json')]

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

    # def collect_openpose_data(self, openpose_output_path, in_channels=2, max_people=3):
    #     def collect_data(lst):
    #         k = np.array([float(c) for c in lst])
    #         x = np.round(k[::3], 8)
    #         y = np.round(k[1::3], 8)
    #         c = np.round(k[2::3], 8)
    #         return np.concatenate((x, y), axis=1), c
    #
    #     file_names = [path.join(openpose_output_path, f) for f in os.listdir(openpose_output_path) if osp.isfile(path.join(openpose_output_path, f)) and f.endswith('json')]
    #
    #     kps = np.zeros((max_people, len(file_names), len(self.layout), in_channels))
    #     scores = np.zeros((max_people, len(file_names), len(self.layout)))
    #
    #     for i, file in tqdm(enumerate(file_names), ascii=True, desc='Collect OpenPose'):
    #         skeletons = read_json(file)['people']
    #         for j, skeleton in enumerate(skeletons):
    #             kp, s = collect_data(skeleton['pose_keypoints_2d'])
    #             kps[j, i, :, :] = kp
    #             scores[j, i, :] = s
    #     return kps, scores
    #
    # def prepare_skeleton_new(self, src_path, result_skeleton_dir=None, source_type=SkeletonSource.VIDEO, out_name=None, label=None, label_index=None):
    #     basename = osp.basename(src_path)
    #     basename_no_ext = osp.splitext(basename)[0] if source_type == SkeletonSource.VIDEO else basename
    #     openpose_output_path = osp.join(self.open_pose_path, 'runs', basename_no_ext) if result_skeleton_dir is None else osp.join(result_skeleton_dir, 'openpose', basename_no_ext)
    #
    #     try:
    #         resolution, fps, frame_count = get_video_properties(src_path)
    #         self._exec_openpose(src_path, openpose_output_path, source_type=source_type)
    #         kp, s = self.collect_openpose_data(openpose_output_path)
    #         skeleton = {
    #             'keypoint': kp,
    #             'keypoint_score': s,
    #             'frame_dir': basename,
    #             'img_shape': resolution,
    #             'original_shape': resolution,
    #             'fps': fps,
    #             'total_frames': frame_count,
    #         }
    #         if label is not None and label_index is not None:
    #             skeleton['label_name'] = label
    #             skeleton['label'] = label_index
    #         if result_skeleton_dir:
    #             result_path = osp.join(result_skeleton_dir, out_name if out_name else f'{basename_no_ext}.json')
    #             write_pkl(skeleton, result_path)
    #         return skeleton
    #     except Exception as e:
    #         print(f'Error creating skeleton from {src_path}: {e}')
    #         raise e
    #     finally:
    #         shutil.rmtree(openpose_output_path)

    def normalize(self, skeleton, centralize):
        new_skeleton = skeleton.copy()
        new_skeleton['data'] = normalize_json(skeleton['data'], skeleton['resolution'], centralize=centralize)
        return new_skeleton

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
        for i, frame_info in tqdm(enumerate(skeleton['data']), ascii=True, desc='To numpy'):
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

    def _to_posec3d_numpy(self, skeleton_data, in_layout, out_layout):
        keypoints = np.zeros((self.num_person_out, len(skeleton_data), len(out_layout), self.C - 1))
        scores = np.zeros((self.num_person_out, len(skeleton_data), len(out_layout)))

        for i, frame_info in enumerate(skeleton_data):
            skeletons = sorted(frame_info['skeleton'], key=lambda s: np.mean(s['pose_score']), reverse=True)[:self.num_person_out]
            for j, skeleton in enumerate(skeletons):
                keypoint, score = np.array([skeleton['pose'][::2], skeleton['pose'][1::2]]).T, np.array(skeleton['pose_score'])
                keypoints[j, i, :, :] = convert_layout(keypoint, in_layout, out_layout)
                scores[j, i, :] = convert_layout(score, in_layout, out_layout)
        return keypoints, scores

    def to_poseC3D(self, json_file, label=None, label_index=None, in_layout=BODY_25_LAYOUT, out_layout=COCO_LAYOUT):
        kp, s = self._to_posec3d_numpy(json_file['data'], in_layout, out_layout)

        result = {
            'keypoint': kp,
            'keypoint_score': s,
            'frame_dir': json_file['name'],
            'video_path': json_file['video_path'],
            'img_shape': json_file['resolution'],
            'original_shape': json_file['resolution'],
            'fps': json_file['fps'],
            'length_seconds': json_file['length_seconds'],
            'frame_count': json_file['frame_count'],
            'adjust': json_file['adjust'],
            'total_frames': len(json_file['data']),
        }
        if label is not None and label_index is not None:
            result['label_name'] = label
            result['label'] = label_index
        return result
