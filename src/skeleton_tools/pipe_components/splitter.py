import numpy as np
from tqdm import tqdm
from os import path

from skeleton_tools.utils.tools import read_json, write_json


class Splitter:
    def __init__(self, step_size, window_size):
        self.step_size = step_size
        self.window_size = window_size

    def split(self, skeleton):
        skeleton_data = skeleton['data']
        T = len(skeleton_data)
        segments = range(0, T - self.window_size + self.step_size, self.step_size)
        result = []
        for s in tqdm(segments, ascii=True, desc='Splitting'):
            t = np.min((s + self.window_size, T))
            window = skeleton_data[s:t]
            for i, w in enumerate(window):
                w['frame_index'] = i
            result.append(window)
            s += self.step_size
        return result

