import os

import matplotlib.colors as mcolors
from os import path as osp
import pandas as pd

from skeleton_tools.openpose_layouts.body import BODY_25_LAYOUT
from skeleton_tools.openpose_layouts.face import FACE_LAYOUT
from skeleton_tools.openpose_layouts.hand import HAND_LAYOUT

NET_NAME = 'JORDI'
NET_FULLNAME = 'Joint Observation RRB Deep-learning Instrument'
REMOTE_STORAGE = r'Z:'

OPENPOSE_ROOT = r'D:\repos\openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended'
MMACTION_ROOT = r'D:\repos\mmaction2'
MMLAB_ENV_PATH = r'C:\Users\owner\anaconda3\envs\open-mmlab\python.exe'
DB_PATH = osp.join(REMOTE_STORAGE, r'recordings')
REDCAP_PATH = osp.join(REMOTE_STORAGE, r'recordings\redcap_db_230906.csv')
ANNOTATIONS_PATH = osp.join(REMOTE_STORAGE, r'Users\TalBarami\lancet_submission_data\annotations\labels.csv')

REAL_DATA_MOVEMENTS = ['Hand flapping', 'Tapping', 'Clapping', 'Fingers', 'Body rocking',
                       'Tremor', 'Spinning in circle', 'Toe walking', 'Back and forth', 'Head movement',
                       'Playing with object', 'Jumping in place', 'Legs movement', 'Feeling texture', 'Other', 'NoAction']

EMOTION_COLS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
LANDMARK_COLS = [i for s in list(zip([f'x_{i}' for i in range(68)], [f'y_{i}' for i in range(68)])) for i in s]
FACE_COLS = ['FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight', 'FaceScore']
ROTATION_COLS = ['Pitch', 'Roll', 'Yaw']

JSON_SOURCES = [
    {'name': 'pose', 'openpose': 'pose_keypoints_2d', 'layout': BODY_25_LAYOUT},
    {'name': 'face', 'openpose': 'face_keypoints_2d', 'layout': FACE_LAYOUT},
    {'name': 'hand_left', 'openpose': 'hand_left_keypoints_2d', 'layout': HAND_LAYOUT},
    {'name': 'hand_right', 'openpose': 'hand_right_keypoints_2d', 'layout': HAND_LAYOUT}
]

COLORS = [{'name': k.split(':')[1], 'value': tuple(int(v.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))} for k, v in mcolors.TABLEAU_COLORS.items()]

LENGTH = 200
MIN_LENGTH = 60
STEP_SIZE = 30
EPSILON = 1e-4
pd.options.display.max_columns = 99

def read_db(file_path=None):
    if file_path is None:
        dbs = [f for f in os.listdir(DB_PATH) if f[:3] == 'db_' and f.endswith('.csv')]
        _db = pd.read_csv(osp.join(DB_PATH, sorted(dbs, key=lambda x: x.split('_')[-1].split('.')[0])[-1]))
    else:
        _db = pd.read_csv(file_path)
    scores = pd.read_csv(REDCAP_PATH, parse_dates=['date_of_birth', 'assessment_date'], infer_datetime_format=True)
    scores['age_days'] = scores['assessment_date'] - scores['date_of_birth']
    scores['age_years'] = scores['age_days'].dt.days / 365.25
    db = pd.merge(_db, scores[[c for c in scores.columns if c not in _db.columns or c == 'assessment']], on='assessment', how='left').drop_duplicates(subset=['basename'])
    return db