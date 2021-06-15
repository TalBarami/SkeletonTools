import numpy as np
from tqdm import tqdm
from os import path

from skeleton_tools.utils.tools import read_json, write_json


class Segmentor:
    def __init__(self, step_size, window_size):
        self.step_size = step_size
        self.window_size = window_size

    def segmentize(self, skeleton_path, out_dir):
        skeleton_data = read_json(skeleton_path)
        basename = path.splitext(path.basename(skeleton_path))[0]

        T = len(skeleton_data)
        segments = range(0, T - self.window_size + self.step_size, self.step_size)

        for s in tqdm(segments, ascii=True, desc='Segmentizing'):
            t = np.min((s + self.window_size, T))
            window = skeleton_data[s:t]
            for i, w in enumerate(window):
                w['frame_index'] = i
            write_json(window, path.join(out_dir, f'{basename}_{s}_{t}.json'))
            s += self.step_size


