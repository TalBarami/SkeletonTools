import pickle

import numpy as np
import torch

from skeleton_tools.utils import skeleton_utils
from skeleton_tools.utils.tools import read_pkl


class SkeletonFeeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """
    def __init__(self,
                 data_path,
                 label_path,
                 random_repetitions=False,
                 random_position=False,
                 random_mirror=False,
                 random_reverse=False,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_repetitions = random_repetitions
        self.random_positioning = random_position
        self.random_mirror = random_mirror
        self.random_reverse = random_reverse
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size

        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        self.sample_name, self.label = read_pkl(self.label_path)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        if self.random_repetitions:
            data_numpy, label = skeleton_utils.random_repetition(data_numpy, self.random_repetitions)

        # processing
        if self.random_mirror and np.random.rand() > 0.5:
            data_numpy = skeleton_utils.mirror_sample(data_numpy)
        if self.random_reverse and np.random.rand() > 0.5:
            data_numpy = skeleton_utils.reverse_sample(data_numpy)
        if self.random_positioning:
            data_numpy = skeleton_utils.random_positioning(data_numpy)

        if self.random_choose:
            data_numpy = skeleton_utils.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = skeleton_utils.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = skeleton_utils.random_move(data_numpy)

        return data_numpy, label
