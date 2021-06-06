import numpy as np

from openpose_layouts.body import BODY_25_LAYOUT
from openpose_layouts.face import FACE_LAYOUT
from openpose_layouts.hand import HAND_LAYOUT


class OpenPoseLayout:
    def __init__(self, name, center, joints, pairs):
        self.name = name
        self.center = center
        self._pose_joints = joints
        self._pose_pairs = pairs
        self._len = len(joints)
        self._pose_map = {}
        for k, v in joints.items():
            self._pose_map[k] = v
            self._pose_map[v] = k

    def __len__(self):
        return self._len

    def joint(self, i):
        return self._pose_map[i]

    def joints(self):
        return self._pose_joints

    def pairs(self):
        return self._pose_pairs


def combine_layouts(layouts):
    name = 'COMBINED'
    center = layouts[0].center
    joints = {}
    pairs = []

    last_index = 0
    for n, l in layouts.items():
        joints |= {k + last_index: f'{n}_{v}' for (k, v) in l.joints()}
        pairs += [(x1 + last_index, x2 + last_index) for (x1, x2) in l.pairs()]
        last_index += len(l)

    layout = OpenPoseLayout(name, center, joints, pairs)
    pairs += [(layout.joint('body_LWrist'), layout.joint('left_hand_root')), (layout.joint('body_RWrist'), layout.joint('right_hand_root'))]

    return layout


COMBINED_BODY25_LAYOUT = combine_layouts({'body': BODY_25_LAYOUT, 'face': FACE_LAYOUT, 'left_hand': HAND_LAYOUT, 'right_hand': HAND_LAYOUT})
