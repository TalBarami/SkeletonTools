import os
from os import path

import matplotlib.pyplot as plt
import numpy as np
# import torch.fft as fft
from scipy.fftpack import fft
import pandas as pd
import seaborn as sns
from sklearn import metrics

from skeleton_tools.openpose_layouts.body import BODY_25_LAYOUT
from skeleton_tools.utils.constants import REAL_DATA_MOVEMENTS
from skeleton_tools.utils.tools import read_json, get_video_properties


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


def bar_plot_lenghts(df_path):
    fig, ax = plt.subplots()
    df = pd.read_csv(df_path)
    df['length'] = df['end_time'] - df['start_time']
    classes, lengths = list(zip(*list(df.groupby('movement')['length'])))
    lengths = [l.to_numpy() for l in lengths]
    groups = list(zip([(0, 2), (2, 5), (5, 8), (8, np.inf)], ['#1D2F6F', '#8390FA', '#6EAF46', '#FAC748']))
    prev = np.zeros(len(classes))
    for (min_len, max_len), color in groups:
        counts = [np.count_nonzero(l[(min_len < l) & (l <= max_len)]) for l in lengths]
        ax.barh(classes, counts, color=color, left=prev)
        prev += counts
    legend = [f'{min_len}s - {max_len}s' for (min_len, max_len), _ in groups]
    legend[-1] = f'{groups[-1][0][0]}s $\leq$'
    plt.legend(legend, loc='lower right', ncol=4)
    plt.xlabel('Number of videos')
    plt.ylabel('Class')
    plt.savefig('resources/figs/videos_lengths.png', bbox_inches='tight')
    plt.show()


def bar_plot_unique_children(df_path):
    fig, ax = plt.subplots()
    df = pd.read_csv(df_path)
    df['ckey'] = df['video'].apply(lambda s: s.split('_')[0])
    gp = df.groupby(['movement'])['ckey'].nunique()
    ax.barh(gp.index, gp.values / df['ckey'].nunique() * 100)
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        ax.annotate(f'{int(width)}%', (x + width + 4, y + height * 0.25), ha='center')
    ax.set_xlim(0, 80)
    ax.set_xlabel('Percentage of unique children')
    ax.set_ylabel('Class')
    plt.savefig('resources/figs/unique_children.png', bbox_inches='tight')
    plt.show()

classmap = {0: 'Hand flapping',
                1: 'Tapping',
                2: 'Clapping',
                3: 'Fingers',
                4: 'Body rocking',
                5: 'Tremor',
                6: 'Spinning in circle',
                7: 'Toe walking',
                8: 'Back and forth',
                9: 'Head movement',
                10: 'Playing with object',
                11: 'Jumping in place'}
classmap = list(classmap.values())

def plot_scores_heatmap(preds_df):
    v = []
    for i in range(len(classmap)):
        df = preds_df[preds_df['y'] == i][preds_df.columns[1:]]
        v.append(df.mean().to_numpy())
    v = np.round(np.array(v), 3)
    df_cm = pd.DataFrame(v, index=classmap, columns=classmap)
    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(df_cm, annot=True)
    ax.figure.tight_layout()
    plt.savefig(r'resources/figs/heatmap.png')
    plt.show()

def plot_conf_matrix(preds_df, norm=False):
    y_hat = preds_df['y']
    y_pred = np.argmax(preds_df[preds_df.columns[1:]].to_numpy(), axis=1)
    cm = metrics.confusion_matrix(y_hat, y_pred)
    if norm:
        counts = preds_df.groupby('y')['0'].count().to_numpy()
        cm = np.round(cm / counts, 3)
    df_cm = pd.DataFrame(cm, index=classmap, columns=classmap)
    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(df_cm, annot=True, fmt='g')
    ax.figure.tight_layout()
    plt.savefig(r'resources/figs/confusion_matrix.png')
    plt.show()

import cv2
if __name__ == '__main__':
    r = r'F:\ALOTAF\2021- All\7'

    cams = [path.join(r, f) for f in os.listdir(r) if path.isdir(path.join(r, f))]
    out = {}
    for cam in cams:
        print(f'Displaying for camera: {path.basename(cam)}')
        c = len(os.listdir(cam))
        results = []
        for file in os.listdir(cam):
            res, fps, length = get_video_properties(path.join(cam, file))
            if fps != np.inf:
                results.append(file)
            print(file, fps)
        out[path.basename(cam)] = [len(results), c, results]
    print(1)

    df = pd.read_csv(r'E:\mmaction2\test.csv')
    label = df['y']
    predict = np.argmax(df[df.columns[1:]].to_numpy(), axis=1)
    result = {s: ((predict == i) & (label == i)).sum() / (label == i).sum() for i, s in enumerate(REAL_DATA_MOVEMENTS)}
    for k, v in result.items():
        print(f'{k} & {np.round(v, 2)}\\% \\\\')
    plot_conf_matrix(df, norm=True)
    plot_scores_heatmap(df)

    # vinfos = []
    # for root, dirs, files in os.walk(r'D:\datasets\tagging_hadas&dor\raw_videos'):
    #     for f in files:
    #         vinfos.append(get_video_properties(path.join(root, f)))
    # resolution = set([v[0] for v in vinfos])
    # fps = np.unique([v[1] for v in vinfos])
    # df_path = r'C:\Users\owner\PycharmProjects\RepetitiveMotionRecognition\resources\labels_complete_updated.csv'
    # bar_plot_lenghts(df_path)
    # bar_plot_unique_children(df_path)
    # skeletons_root = r'D:\datasets\autism_center\skeletons_filtered\data'
    # videos_root = r'D:\datasets\autism_center\segmented_videos'
    #
    # files = [path.splitext(f)[0] for f in os.listdir(videos_root) if f.endswith('.avi')]
    # files = [f for i, f in enumerate(files) if f'{f}.json' in set(os.listdir(skeletons_root))]
    # names = []
    #
    #
    #
    #
    # for label in ['Hand flapping', 'Tapping', 'Clapping', 'Body rocking',
    #               'Tremor', 'Toe walking', 'Head movement',
    #               'Playing with object', 'Jumping in place', 'NoAction']:
    #     sub = [f for f in files if label in f]
    #     for filename in sub:
    #         data = read_json(path.join(skeletons_root, f'{filename}.json'))['data']
    #         if len(data) < 40:
    #             print(f'Video is too short: {filename}')
    #             continue
    #         np_result = np.zeros((3, 25, len(data)))
    #         for frame_info in data:
    #             if frame_info['skeleton']:
    #                 skel = frame_info['skeleton'][0]
    #                 np_result[0, :, frame_info['frame_index']] = skel['pose'][0::2]
    #                 np_result[1, :, frame_info['frame_index']] = skel['pose'][1::2]
    #                 np_result[2, :, frame_info['frame_index']] = skel['pose_score']
    #         plot_fft(np_result, label, f'{filename}')
    #     break
