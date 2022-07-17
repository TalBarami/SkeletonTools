import matplotlib.colors as mcolors
from os import path as osp

from skeleton_tools.openpose_layouts.body import BODY_25_LAYOUT
from skeleton_tools.openpose_layouts.face import FACE_LAYOUT
from skeleton_tools.openpose_layouts.hand import HAND_LAYOUT

NET_NAME = 'JORDI'
NET_FULLNAME = 'Joint Observation RRB Deep-learning Instrument'
REMOTE_STORAGE = osp.join(r'\\ac-s1', 'Data', 'Autism Center')

OPENPOSE_ROOT = r'E:\research\openpose'
MMACTION_ROOT = r'E:\research\mmaction2'
MMLAB_ENV_PATH = r'C:\Users\owner\anaconda3\envs\open-mmlab\python.exe'
DB_PATH = osp.join(REMOTE_STORAGE, r'Users\TalBarami\NAS_database_final.csv')
ANNOTATIONS_PATH = osp.join(REMOTE_STORAGE, r'Users\TalBarami\lancet_submission_data\annotations\labels.csv')

REAL_DATA_MOVEMENTS = ['Hand flapping', 'Tapping', 'Clapping', 'Fingers', 'Body rocking',
                       'Tremor', 'Spinning in circle', 'Toe walking', 'Back and forth', 'Head movement',
                       'Playing with object', 'Jumping in place', 'Legs movement', 'Feeling texture', 'Other', 'NoAction']
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
