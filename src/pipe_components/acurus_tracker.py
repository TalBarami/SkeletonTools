import os
import shlex
import shutil
import subprocess
from itertools import chain
from os import path
from pathlib import Path

import numpy as np
from tqdm import tqdm

from SkeletonTools.src.openpose_layouts.openpose_layout import OpenPoseLayout
from SkeletonTools.src.utils.tools import read_json, write_json


class AcurusTracker:
    def __init__(self, skeleton_layout: OpenPoseLayout, acurus_path='C:/research/AcurusTrack'):
        self.skeleton_layout = skeleton_layout
        self.acurus_path = acurus_path
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

    def track(self, skeleton, resolution):
        process_dir = path.join(self.acurus_path, 'process')
        Path(process_dir, 'acurus').mkdir(parents=True, exist_ok=True)
        pre_processed_path = path.join(process_dir, 'acurus', 'pre.json')
        write_json(self.skeleton_to_acurus(skeleton), pre_processed_path)
        width, height = resolution
        exp_name = f'exp_v'
        save_dir = path.join(process_dir, 'acurus_results')

        params = {
            'detections': f'\"{pre_processed_path}\"',
            'width': width,
            'height': height,
            'video_name': f'\"v\"',
            'exp_name': f'\"{exp_name}\"',
            'save_dir': f'\"{save_dir}\"',
            'force': False
        }

        args = ' '.join([f'--{k} {v}' for k, v in params.items()])

        cmd = f'python {path.join(acurus_path, "run.py")} {args}'.replace('\\', '/')
        print(f'Executing: {cmd}')
        cwd = os.getcwd()
        os.chdir(acurus_path)
        try:
            subprocess.check_call(shlex.split(cmd), universal_newlines=True)
        except:
            params['force'] = True
            args = ' '.join([f'--{k} {v}' for k, v in params.items()])
            cmd = f'python {path.join(acurus_path, "run.py")} {args}'.replace('\\', '/')
            print(f'Execution failed, trying with force: {cmd}')
            try:
                subprocess.check_call(shlex.split(cmd), universal_newlines=True)
            except:
                print(f'Execution with force failed: {cmd}')
                return skeleton
        finally:
            os.chdir(cwd)

        final_meta = read_json(path.join(save_dir, 'v', exp_name, 'result', 'result.json'))
        shutil.rmtree(process_dir)
        return self.acurus_to_skeleton(final_meta)
