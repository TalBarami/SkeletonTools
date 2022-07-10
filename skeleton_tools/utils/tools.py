import json
import os
import pickle
import subprocess
from json import JSONDecodeError
from os import path as osp
from pathlib import Path

import ffmpeg
import pandas as pd
import numpy as np
import cv2

from skeleton_tools.utils.constants import REMOTE_STORAGE

def init_logged(log_path='resources/log.txt'):
    logging.basicConfig(filename=log_path, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info('Initialization Success')

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


def get_video_properties(filename):
    try:
        vinf = ffmpeg.probe(filename)

        resolution_candidates = [(vinf['streams'][i]['width'], vinf['streams'][i]['height']) for i in range(len(vinf['streams'])) if 'width' in vinf['streams'][i].keys() and 'height' in vinf['streams'][i].keys()]
        fps_candidates = [vinf['streams'][i]['avg_frame_rate'] for i in range(len(vinf['streams'])) if 'avg_frame_rate' in vinf['streams'][i].keys()] + \
                         [vinf['streams'][i]['r_frame_rate'] for i in range(len(vinf['streams'])) if 'r_frame_rate' in vinf['streams'][i].keys()]
        fps_candidates = [x for x in fps_candidates if x != '0/0']

        resolution = resolution_candidates[0] if len(resolution_candidates) > 0 else None
        fps = eval(fps_candidates[0]) if len(fps_candidates) > 0 else None
        length = eval(vinf['format']['duration']) if 'format' in vinf.keys() and 'duration' in vinf['format'].keys() else None
        frame_count = length * fps if length and fps else None
        return resolution, fps, frame_count, length
    except Exception as e:
        return None, None, None, None


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
