import json
import pickle

import torch
from numpy.lib.format import open_memmap
import numpy as np

import os
from os import path

from skeleton_tools.openpose_layouts.body import BODY_25_LAYOUT
from skeleton_tools.pipe_components.openpose_initializer import OpenposeInitializer
from skeleton_tools.utils.constants import LENGTH
from skeleton_tools.utils.tools import read_json, write_pkl


def gendata(
        data_path,
        label_path,
        data_out_path,
        label_out_path,
        in_channels=3,
        layout=BODY_25_LAYOUT,
        num_person_in=5,
        num_person_out=3,
        max_frame=LENGTH):

    initializer = OpenposeInitializer(layout, in_channels=in_channels, length=max_frame, num_person_in=num_person_in, num_person_out=num_person_out)

    n_joints = len(layout)
    files = [f for f in os.listdir(data_path)]
    labels_json = read_json(label_path)
    labels_out = []

    fp = open_memmap(data_out_path,
                     dtype='float32',
                     mode='w+',
                     shape=(len(files), in_channels, max_frame, n_joints,
                            num_person_out))

    for i, file in enumerate(files):
        json_data = read_json(path.join(data_path, file))
        data = initializer.to_numpy(json_data)
        fp[i, :, 0:data.shape[1], :, :] = data

        basename = path.splitext(file)[0]
        label = labels_json[basename]['label_index']
        labels_out.append(label)

    write_pkl((files, list(labels_out)), label_out_path)

# class JsonDataset(torch.utils.data.Dataset):
#     """ Feeder for skeleton-based action recognition in kinetics-skeleton dataset
#     Arguments:
#         data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
#         label_path: the path to label
#         random_choose: If true, randomly choose a portion of the input sequence
#         random_shift: If true, randomly pad zeros at the begining or end of sequence
#         random_move: If true, perform randomly but continuously changed transformation to input sequence
#         window_size: The length of the output sequence
#         pose_matching: If ture, match the pose between two frames
#         num_person_in: The number of people the feeder can observe in the input sequence
#         num_person_out: The number of people the feeder in the output sequence
#         debug: If true, only use the first 100 samples
#     """
#     def __init__(self,
#                  data_path,
#                  label_path,
#                  ignore_empty_sample=True,
#                  random_choose=False,
#                  random_shift=False,
#                  random_move=False,
#                  random_repetition=False,
#                  window_size=-1,
#                  pose_matching=False,
#                  num_person_in=5,
#                  num_person_out=2,
#                  n_joints=21,
#                  debug=False):
#         self.debug = debug
#         self.data_path = data_path
#         self.label_path = label_path
#         self.random_choose = random_choose
#         self.random_shift = random_shift
#         self.random_move = random_move
#         self.random_repetition = random_repetition
#         self.window_size = window_size
#         self.num_person_in = num_person_in
#         self.num_person_out = num_person_out
#         self.n_joints = n_joints
#         self.pose_matching = pose_matching
#         self.ignore_empty_sample = ignore_empty_sample
#
#         self.load_data()
#
#     def load_data(self):
#         # load file list
#         self.sample_name = os.listdir(self.data_path)
#
#         if self.debug:
#             self.sample_name = self.sample_name[0:2]
#
#         # load label
#         label_path = self.label_path
#         with open(label_path) as f:
#             label_info = json.load(f)
#
#         sample_id = [name.split('.')[0] for name in self.sample_name]
#         self.label = np.array(
#             [label_info[id]['label_index'] for id in sample_id])
#         has_skeleton = np.array(
#             [label_info[id]['has_skeleton'] for id in sample_id])
#
#         # ignore the samples which does not has skeleton sequence
#         if self.ignore_empty_sample:
#             self.sample_name = [
#                 s for h, s in zip(has_skeleton, self.sample_name) if h
#             ]
#             self.label = self.label[has_skeleton]
#
#         # output data shape (N, C, T, V, M)
#         self.N = len(self.sample_name)  #sample
#         self.C = 3  #channel
#         self.T = self.window_size  #frame
#         self.V = self.n_joints  #joint
#         self.M = self.num_person_out  #person
#
#     def __len__(self):
#         return len(self.sample_name)
#
#     def __iter__(self):
#         return self
#
#     def __getitem__(self, index):
#
#         # output shape (C, T, V, M)
#         # get data
#         sample_name = self.sample_name[index]
#         sample_path = os.path.join(self.data_path, sample_name)
#         with open(sample_path, 'r') as f:
#             video_info = json.load(f)
#
#         # fill data_numpy
#         data_numpy = np.zeros((self.C, self.T, self.V, self.num_person_in))
#         for frame_info in video_info['data']:
#             frame_index = int(frame_info['frame_index'])
#             if frame_index == self.T:
#                 break
#             for m, skeleton_info in enumerate(frame_info["skeleton"]):  # TODO: case when id > num_person_in, but len(skeleton) < num_person_in
#                 pid = m if ('person_id' not in skeleton_info.keys()) else skeleton_info['person_id']
#                 if type(pid) == list:
#                     pid = m if pid[0] < 0 else pid[0]
#                 pid %= self.num_person_in
#                 pose = skeleton_info['pose']
#                 score = skeleton_info['score'] if 'score' in skeleton_info.keys() else skeleton_info['pose_score']
#                 data_numpy[0, frame_index, :, pid] = pose[0::2]
#                 data_numpy[1, frame_index, :, pid] = pose[1::2]
#                 data_numpy[2, frame_index, :, pid] = score
#
#         # centralization
#         data_numpy[0:2] = data_numpy[0:2] - 0.5
#         data_numpy[0][data_numpy[2] == 0] = 0
#         data_numpy[1][data_numpy[2] == 0] = 0
#
#         # get & check label index
#         label = video_info['label_index']
#         assert (self.label[index] == label)
#
#         # data augmentation
#         if self.random_shift:
#             data_numpy = skeleton_utils.random_shift(data_numpy)
#         if self.random_choose:
#             data_numpy = skeleton_utils.random_choose(data_numpy, self.window_size)
#         elif self.window_size > 0:
#             data_numpy = skeleton_utils.auto_pading(data_numpy, self.window_size)
#         if self.random_move:
#             data_numpy = skeleton_utils.random_move(data_numpy)
#         if self.random_repetition:
#             data_numpy, label = skeleton_utils.random_repetition(data_numpy)
#             self.label[index] = label
#
#         # sort by score
#         sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
#         for t, s in enumerate(sort_index):
#             data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2, 0))
#         data_numpy = data_numpy[:, :, :, 0:self.num_person_out]
#
#         # match poses between 2 frames
#         if self.pose_matching:
#             data_numpy = skeleton_utils.openpose_match(data_numpy)
#
#         return data_numpy, label
#
#     def top_k(self, score, top_k):
#         assert (all(self.label >= 0))
#
#         rank = score.argsort()
#         hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
#         return sum(hit_top_k) * 1.0 / len(hit_top_k)
#
#     def top_k_by_category(self, score, top_k):
#         assert (all(self.label >= 0))
#         return tools.top_k_by_category(self.label, score, top_k)
#
#     def calculate_recall_precision(self, score):
#         assert (all(self.label >= 0))
#         return tools.calculate_recall_precision(self.label, score)