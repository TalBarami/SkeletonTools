import os
from os import path

import matplotlib.pyplot as plt
import numpy as np
# import torch.fft as fft
from scipy.fftpack import fft
import pandas as pd


from skeleton_tools.openpose_layouts.body import BODY_25_LAYOUT
from skeleton_tools.utils.tools import read_json


def plot_fft(data_numpy, title, filename):
    C, J, N = data_numpy.shape
    rJ = int(np.sqrt(J))

    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle(title)
    legend = ['x', 'y']
    subfigs = fig.subfigures(rJ, rJ)

    x = np.arange(N)
    xf = x[1:N // 2]
    for j, subfig in enumerate(subfigs.flat):
        subfig.suptitle(BODY_25_LAYOUT.joint(j))
        axs = subfig.subplots(2, 1)
        scores = data_numpy[2, j, :]
        for c in range(2):
            y = data_numpy[c, j, :].copy()
            if (scores < 0.4).mean() > 0.3:
                y[:] = 0
            else:
                s = pd.Series(y)
                s[scores < 0.4] = np.nan
                y = s.interpolate(method='spline', order=2, limit_direction='both').to_numpy()

            yf = fft(y)[1:N // 2]
            axs[0].set_title('org space', fontsize='small')
            axs[1].set_title('fft space', fontsize='small')
            axs[0].plot(x, y, label=legend[c])
            axs[1].plot(xf, np.abs(yf), label=legend[c])
            axs[1].axvline(7, color='r')
            axs[1].axvline(20, color='r')
    dst_path = r'D:\datasets\autism_center\fftfigs'
    fig.savefig(path.join(dst_path, f'{filename}.png'), dpi=100)
    # plt.show()


def plot_layers():
    return

if __name__ == '__main__':
    skeletons_root = r'D:\datasets\autism_center\skeletons_filtered\data'
    videos_root = r'D:\datasets\autism_center\segmented_videos'

    files = [path.splitext(f)[0] for f in os.listdir(videos_root) if f.endswith('.avi')]
    files = [f for i, f in enumerate(files) if f'{f}.json' in set(os.listdir(skeletons_root))]
    names = []




    for label in ['Hand flapping', 'Tapping', 'Clapping', 'Body rocking',
                  'Tremor', 'Toe walking', 'Head movement',
                  'Playing with object', 'Jumping in place', 'NoAction']:
        sub = [f for f in files if label in f]
        for filename in sub:
            data = read_json(path.join(skeletons_root, f'{filename}.json'))['data']
            if len(data) < 40:
                print(f'Video is too short: {filename}')
                continue
            np_result = np.zeros((3, 25, len(data)))
            for frame_info in data:
                if frame_info['skeleton']:
                    skel = frame_info['skeleton'][0]
                    np_result[0, :, frame_info['frame_index']] = skel['pose'][0::2]
                    np_result[1, :, frame_info['frame_index']] = skel['pose'][1::2]
                    np_result[2, :, frame_info['frame_index']] = skel['pose_score']
            plot_fft(np_result, label, f'{filename}')
        break