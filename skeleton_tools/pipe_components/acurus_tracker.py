import os
import shlex
import shutil
import subprocess
from itertools import chain
from os import path
from pathlib import Path

import cv2
import numpy as np
from evenvizion.processing import video_processing, json
from tqdm import tqdm

from skeleton_tools.openpose_layouts.graph_layout import GraphLayout
from skeleton_tools.utils.tools import read_json, write_json


class AcurusTracker:
    def __init__(self, skeleton_layout: GraphLayout, acurus_path='C:/research/AcurusTrack', acurus_env=r'C:\Users\owner\anaconda3\envs\acurus\python.exe'):
        self.skeleton_layout = skeleton_layout
        self.acurus_path = acurus_path
        self.acurus_env = acurus_env
        self.special_joints = {
            'MidHip': [self.skeleton_layout.joint('MidHip')],
            'Neck': [self.skeleton_layout.joint('Neck')],
            'Nose': [self.skeleton_layout.joint('Nose')],
            'BigToes': [self.skeleton_layout.joint('LBigToe'), self.skeleton_layout.joint('RBigToe')]
        }

    def special_joints_info(self, name, x, y, c, idxs):
        return {
            f'{name}_x': np.mean([x[i] for i in idxs]),
            f'{name}_y': np.mean([y[i] for i in idxs]),
            f'{name}_score': np.mean([c[i] for i in idxs]),
        }

    def openpose_to_acurus(self, openpose_dir):
        file_names = [path.join(openpose_dir, f) for f in os.listdir(openpose_dir) if path.isfile(path.join(openpose_dir, f)) and f.endswith('json')]

        result = {}
        for frame, file in tqdm(enumerate(file_names), ascii=True, desc="Openpose to Acurus"):
            skeletons = []
            json = read_json(file)
            people = json['people']
            for p in people:
                k = np.array([float(c) for c in p['pose_keypoints_2d']])
                x = np.round(k[::3], 8)
                y = np.round(k[1::3], 8)
                c = np.round(k[2::3], 8)
                p_stats = {'person': [list(p) for p in zip(x, y, c)]}
                for n, j in self.special_joints.items():
                    p_stats.update(self.special_joints_info(n, x, y, c, j))

                skeletons.append(p_stats)
            result[frame] = skeletons
        return result

    def skeleton_to_acurus(self, skeleton):
        result = {}
        for frame_id, frame_info in tqdm(enumerate(skeleton), ascii=True, desc="JSON to Acurus"):
            skeletons = []
            people = frame_info['skeleton']
            for p in people:
                x, y = p['pose'][::2], p['pose'][1::2]
                c = p['pose_score']
                p_stats = {'person': [list(p) for p in zip(x, y, c)]}
                for n, j in self.special_joints.items():
                    p_stats.update(self.special_joints_info(n, x, y, c, j))
                skeletons.append(p_stats)
            result[frame_id] = skeletons
        return result

    def acurus_to_skeleton(self, acurus_json):
        result = []
        max_frame = np.max([int(k) for k in acurus_json.keys()]) + 1
        for i in tqdm(range(max_frame), ascii=True, desc="Acurus to JSON"):
            i = str(i)
            skeletons = []
            if i in acurus_json.keys():
                for person in acurus_json[i]:
                    person_id, pose = person['index'], eval(person['person'])
                    skeletons.append({
                        'person_id': person_id,
                        'pose': list(chain(*[(p[0], p[1]) for p in pose])),
                        'pose_score': [p[2] for p in pose]
                    })
            result.append({'frame_index': int(i),
                           'skeleton': skeletons})
        return result

    def track(self, skeleton, video_path, fixed_coordinate=False):
        video_name, _ = path.splitext(path.basename(video_path))
        experiment_name = 'exp'
        result_path = path.join(self.acurus_path, 'results')
        process_dir = path.join(result_path, 'process', video_name)
        Path(process_dir).mkdir(parents=True, exist_ok=True)
        acurus_skeleton_path = path.join(process_dir, 'skeleton.json')
        homography_dict_path = path.join(process_dir, 'homography_dict.json')
        out_path = path.join(result_path, video_name, experiment_name, 'result', 'result.json')

        write_json(self.skeleton_to_acurus(skeleton), acurus_skeleton_path)

        params = {
            'detections': f'\"{acurus_skeleton_path}\"',
            'video_path': f'\"{video_path}\"',
            'video_name': f'\"{video_name}\"',
            'exp_name': f'\"{experiment_name}\"'
        }
        if fixed_coordinate:
            print(f'Creating homography dict...')
            cap = cv2.VideoCapture(video_path)
            homography_dict = video_processing.get_homography_dict(cap)
            with open(homography_dict_path, "w") as json_:
                json.dump(homography_dict, json_)
            params['path_to_homography_dict'] = f'\"{homography_dict_path}\"'
            cap.release()

        # process_dir = path.join(self.acurus_path, 'process')
        # Path(process_dir, 'acurus').mkdir(parents=True, exist_ok=True)
        # pre_processed_path = path.join(process_dir, 'acurus', 'pre.json')
        # write_json(self.skeleton_to_acurus(skeleton), pre_processed_path)
        # exp_name = f'exp_v'
        # save_dir = path.join(process_dir, 'acurus_results')
        #
        # params = {
        #     'detections': f'\"{pre_processed_path}\"',
        #     'width': width,
        #     'height': height,
        #     'video_name': f'\"v\"',
        #     'exp_name': f'\"{exp_name}\"',
        #     'save_dir': f'\"{save_dir}\"',
        #     'force': False
        # }

        # if video_path is not None:
        #     path_to_homography_dict = path.join(process_dir, 'homography_dict.json')
        #     try:
        #         print(f'Creating homography dict...')
        #         cap = cv2.VideoCapture(video_path)
        #         homography_dict = video_processing.get_homography_dict(cap)
        #         with open(path_to_homography_dict, "w") as json_:
        #             json.dump(homography_dict, json_)
        #         params['path_to_homography_dict'] = f'\"{path_to_homography_dict}\"'
        #     finally:
        #         if cap is not None and cap.isOpened():
        #             cap.release()

        args = ' '.join([f'--{k} {v}' for k, v in params.items()])

        cmd = f'{self.acurus_env} {path.join(self.acurus_path, "run.py")} {args}'.replace('\\', '/')
        print(f'Executing: {cmd}')
        cwd = os.getcwd()
        os.chdir(self.acurus_path)
        try:
            subprocess.check_call(shlex.split(cmd), universal_newlines=True)
        # except:
        #     params['force'] = True
        #     args = ' '.join([f'--{k} {v}' for k, v in params.items()])
        #     cmd = f'python {path.join(self.acurus_path, "run.py")} {args}'.replace('\\', '/')
        #     print(f'Execution failed, trying with force: {cmd}')
        #     subprocess.check_call(shlex.split(cmd), universal_newlines=True)
        #     # try:
        #     #     subprocess.check_call(shlex.split(cmd), universal_newlines=True)
        #     # except:
        #     #     print(f'Execution with force failed: {cmd}')
        #     #     return skeleton
        finally:
            os.chdir(cwd)

        final_meta = read_json(out_path)
        shutil.rmtree(process_dir)
        shutil.rmtree(path.join(self.acurus_path, 'results', video_name))
        return self.acurus_to_skeleton(final_meta)
