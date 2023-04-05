import json
import logging
import os
import pickle
import subprocess
from json import JSONDecodeError
from os import path as osp
from pathlib import Path
import itertools

import ffmpeg
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
import cv2

from skeleton_tools.utils.constants import REMOTE_STORAGE
pd.set_option('display.expand_frame_repr', False)

class DataWrapper:
    def __init__(self,  read_func):
        self.read_func = read_func
        self.data = None

    def get(self):
        if self.data is None:
            self.data = self.read_func()
        return self.data

    def __call__(self):
        return self.get()

def scan_db(root=r'Z:\recordings', reload=True):
    load_from = osp.join(REMOTE_STORAGE, 'recordings', 'db.csv')
    if osp.exists(load_from):
        db = pd.read_csv(load_from)
        if not reload:
            return db
    else:
        db = pd.DataFrame(columns=['filename', 'file_path', 'width', 'height', 'fps', 'frame_count', 'length_seconds'])

    for r, d, fs in os.walk(root):
        if 'Asaf' in r or 'Face camera' in r:
            continue
        for f in fs:
            if f in db['filename'].unique():
                continue
            if f.lower().endswith('.mp4') or f.lower().endswith('.avi'):
                try:
                    (width, height), fps, frame_count, length_seconds = get_video_properties(osp.join(r, f))
                    db.loc[db.shape[0]] = [f, osp.join(r, f), width, height, fps, frame_count, length_seconds]
                except Exception as e:
                    print(f'Error extracting information from {f}.')
    db['basename'] = db['filename'].apply(lambda s: osp.splitext(s)[0])
    db['assessment'] = db['basename'].apply(lambda s: '_'.join(s.split('_')[:4]))
    db['child_id'] = db['filename'].apply(lambda s: s.split('_')[0])
    db['type'] = db['assessment'].apply(lambda s: s.split('_')[1])
    db['date'] = db['assessment'].apply(lambda s: s.split('_')[3])
    db.to_csv(load_from, index=False)
    return db

def create_config(dict_conf, out=None):
    for k, v in dict_conf.items():
        if type(v) == str and ('path' in k or 'dir' in k):
            dict_conf[k] = v.replace('\\', '/')
    config = OmegaConf.create(dict_conf)
    with open(out.replace('\\', '/'), 'w') as fp:
        OmegaConf.save(config=config, f=fp.name)
    return config

def init_logger(log_name, log_path=None):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if log_path is not None:
        fh = logging.FileHandler(osp.join(log_path, f'{log_name}.log'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # logging.basicConfig(filename=osp.join(log_path, log_name), level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
    #                     datefmt='%d/%m/%Y %H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler())
    logger.info(f'Initialization Success: {log_name}')
    return logger

def get_videos(root=osp.join(REMOTE_STORAGE, 'recordings')):
    video_files = {}
    for dir_path, dirs, files in os.walk(root):
        for file in files:
            if osp.splitext(file)[1].lower() in ['.mp4', '.avi']:
                video_files[osp.basename(file)] = osp.join(dir_path, file)
            else:
                print(1)
    return video_files


def init_directories(*dirs):
    for dir in dirs:
        Path(dir).mkdir(parents=True, exist_ok=True)


def read_json(file):
    try:
        with open(file, 'rb') as j:
            return json.loads(j.read())
    except (OSError, UnicodeDecodeError, JSONDecodeError) as e:
        print(f'Error while reading {file}: {e}')
        raise e


def write_json(j, dst):
    with open(dst, 'w') as f:
        json.dump(j, f)


def read_pkl(file):
    try:
        with open(file, 'rb') as p:
            return pickle.load(p)
    except (OSError, UnicodeDecodeError, JSONDecodeError) as e:
        print(f'Error while reading {file}: {e}')
        raise e


def write_pkl(p, dst):
    with open(dst, 'wb') as f:
        pickle.dump(p, f)


def take_subclip(video_path, start_time, end_time, fps, out_path):
    ffmpeg.input(video_path).video \
        .trim(start=start_time, end=end_time) \
        .setpts('PTS-STARTPTS') \
        .filter('fps', fps=fps, round='up') \
        .output(out_path) \
        .run()


def get_video_properties(filename, method='ffmpeg'):

    try:
        vinf = ffmpeg.probe(filename)

        resolution_candidates = [(vinf['streams'][i]['width'], vinf['streams'][i]['height']) for i in range(len(vinf['streams'])) if 'width' in vinf['streams'][i].keys() and 'height' in vinf['streams'][i].keys()]
        fps_candidates = [vinf['streams'][i]['avg_frame_rate'] for i in range(len(vinf['streams'])) if 'avg_frame_rate' in vinf['streams'][i].keys()] + \
                         [vinf['streams'][i]['r_frame_rate'] for i in range(len(vinf['streams'])) if 'r_frame_rate' in vinf['streams'][i].keys()]
        fps_candidates = [x for x in fps_candidates if x != '0/0']

        resolution = resolution_candidates[0] if len(resolution_candidates) > 0 else None
        fps = eval(fps_candidates[0]) if len(fps_candidates) > 0 else None
        length_candidates = [vinf['streams'][i]['duration'] for i in range(len(vinf['streams'])) if 'duration' in vinf['streams'][i].keys()]
        if 'format' in vinf.keys() and 'duration' in vinf['format'].keys():
            length_candidates.append(vinf['format']['duration'])
        length = eval(length_candidates[0]) if len(length_candidates) > 0 else None
        if length is not None and fps is not None:
            estimated_frame = length * fps
        frame_candidates = [eval(vinf['streams'][i]['nb_frames']) for i in range(len(vinf['streams'])) if 'nb_frames' in vinf['streams'][i].keys()]
        frame_candidates = [f for f in frame_candidates if np.abs(f - estimated_frame) < np.min((50, estimated_frame * 0.1))]
        frame_count = int(np.max(frame_candidates)) if len(frame_candidates) > 0 else int(np.ceil(length * fps)) if length and fps else None
    except Exception:
        try:
            cap = cv2.VideoCapture(filename)
            resolution = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            length = frame_count / fps
        except Exception as e:
            raise e
        finally:
            cap.release()
    return resolution, fps, frame_count, length


def generate_label_json(skeletons_dir):
    data_dir = osp.join(skeletons_dir, 'data')
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    result = {}
    for file in files:
        name = file.split('.')[0]
        json_file = read_json(osp.join(data_dir, file))
        result[name] = {'has_skeleton': True, 'label': json_file['label'], 'label_index': json_file['label_index']}
    write_json(result, osp.join(skeletons_dir, 'label.json'))


def top_k_by_category(label, score, top_k):
    instance_num, class_num = score.shape
    rank = score.argsort()
    hit_top_k = [[] for i in range(class_num)]
    for i in range(instance_num):
        l = label[i]
        hit_top_k[l].append(l in rank[i, -top_k:])

    accuracy_list = []
    for hit_per_category in hit_top_k:
        if hit_per_category:
            accuracy_list.append(
                sum(hit_per_category) * 1.0 / len(hit_per_category))
        else:
            accuracy_list.append(0.0)
    return accuracy_list


def calculate_recall_precision(label, score):
    instance_num, class_num = score.shape
    rank = score.argsort()
    confusion_matrix = np.zeros([class_num, class_num])

    for i in range(instance_num):
        true_l = label[i]
        pred_l = rank[i, -1]
        confusion_matrix[true_l][pred_l] += 1

    precision = []
    recall = []

    for i in range(class_num):
        true_p = confusion_matrix[i][i]
        false_n = sum(confusion_matrix[i, :]) - true_p
        false_p = sum(confusion_matrix[:, i]) - true_p
        precision.append(true_p * 1.0 / (true_p + false_p))
        recall.append(true_p * 1.0 / (true_p + false_n))

    return precision, recall


def json_to_pandas(file_path):
    json_file = read_json(file_path)
    data = [[n for n in json_file.keys()]] + [x for x in zip(*[x.values() for x in json_file.values()])]
    return pd.DataFrame(np.array(data).T, columns=['name', 'has_skeleton', 'label', 'label_index'])


def combine_meta_data(face_dir, hand_dir, combined_dir):
    files = [(f, osp.join(face_dir, f), osp.join(hand_dir, f)) for f in os.listdir(face_dir)]
    for name, face, hand in files:
        face = read_json(face)
        hand = read_json(hand)
        n = len(face['people'])
        for i in range(n):
            # if not (face['people'][i]['pose_keypoints_2d'] == hand['people'][i]['pose_keypoints_2d']):
            #     print(f'Error: {i}')
            face['people'][i]['hand_left_keypoints_2d'] = hand['people'][i]['hand_left_keypoints_2d']
            face['people'][i]['hand_right_keypoints_2d'] = hand['people'][i]['hand_right_keypoints_2d']
        write_json(face, osp.join(combined_dir, name))

def pairwise(iterable):
    "s -> (s0, s1), (s1, s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

if __name__ == '__main__':
    db = scan_db()
    asmt = db.groupby('assessment').first().reset_index()[['assessment', 'child_id', 'date', 'type']]
    print(1)
    db2 = scan_db(properties=True)
    print(2)