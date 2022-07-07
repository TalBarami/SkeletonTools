import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from os import path

from skeleton_tools.utils.constants import LENGTH, STEP_SIZE, MIN_LENGTH


class Splitter(Dataset):
    def __init__(self, skeleton, sequence_length=LENGTH, step_size=STEP_SIZE, min_length=MIN_LENGTH):
        self.skeleton = skeleton
        self.sequence_length = sequence_length
        self.step_size = step_size
        self.min_length = min_length
        self.N, self.T, self.J, self.C = self.skeleton['keypoint'].shape

        self.intervals = [(x, min(x+self.sequence_length, self.T)) for x in range(0, max(self.T, self.T - self.sequence_length + self.step_size), self.step_size) if (min(x+self.sequence_length, self.T) - x) >= self.min_length]
        self.template = {k: v for k, v in self.skeleton.items() if k not in ['keypoint', 'keypoint_score', 'total_frames', 'frame_dir']}
        self.template['basename'] = self.skeleton['frame_dir']

    def __getitem__(self, index):
        s, t = self.intervals[index]
        kp = self.skeleton['keypoint'][:, s:t, :, :]
        kps = self.skeleton['keypoint_score'][:, s:t, :]
        basename, _ = path.splitext(self.template["basename"])
        return {**dict(self.template), **{'keypoint': kp,
                                          'keypoint_score': kps,
                                          'frame_dir': f'{basename}_{index}',
                                          'total_frames': t-s,
                                          'segment_name': f'{basename}_{s}_{t}',
                                          'start': s,
                                          'end': t,
                                          'label': -1}}

    def __len__(self):
        return len(self.intervals)

    def collect(self):
        return [x for x in self]