import os
from os import path
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
# import torch.fft as fft
from scipy.fftpack import fft
import pandas as pd
import seaborn as sns
from sklearn import metrics
from tqdm import tqdm

from skeleton_tools.openpose_layouts.body import BODY_25_LAYOUT
from skeleton_tools.skeleton_visualization.json_visualizer import JsonVisualizer
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

def export_frames_for_figure():
    def export_frames(vis, skeleton_json, out_path):
        Path(out_path).mkdir(parents=True, exist_ok=True)
        fps, length, (width, height), kp, c, pids = vis.get_video_info(None, skeleton_json)
        for i in tqdm(range(length), desc="Writing video result"):
            if i < len(kp):
                skel_frame = vis.draw_skeletons(np.zeros((height, width, 3), dtype=np.uint8), kp[i], c[i], (width, height), pids[i])
                cv2.imwrite(path.join(out_path, f'{i}.png'), skel_frame)

    root = r'D:\datasets\taggin_hadas&dor_remake'
    files = ['100670545_Linguistic_Clinical_090720_0909_2_Spinning in circle_13543_14028.avi',
             '101614003_338720063_ADOS_191217_1244_1_Spinning in circle_24660_24810.avi',
             '101615086_ADOS_Clinical_030320_1301_2_Head movement_30329_30480.avi',
             '101631802_ADOS_Clinical_050120_1107_3_Body rocking_16633_17179.avi',
             '101631802_ADOS_Clinical_050120_1107_4_Body rocking_16633_17179.avi',
             '101857828_339532814_ADOS_110618_1216_1_Hand flapping_31110_31260.avi',
             '101991079_ADOS_Clinical_020220_0926_4_Head movement_3211_3332.avi',
             '102033871_ADOS_Clinical_191119_1216_1_Hand flapping_89295_89447.avi',
             '102033871_Cognitive_Clinical_041219_1301_4_Hand flapping_84776_84928.avi',
             '102105601_340358720_ADOS_070118_0932_1_Hand flapping_51780_52320.avi']
    v = JsonVisualizer(graph_layout=BODY_25_LAYOUT)
    for f in files:
        name, ext = path.splitext(f)
        j = read_json(path.join(root, 'skeletons', f'{name}.json'))
        # v.create_double_frame_video(path.join(root, 'segmented_videos', f), j, f'C:/Users/owner/Desktop/out/{name}2{ext}')
        export_frames(v, j, f'C:/Users/owner/Desktop/out/{name}')


if __name__ == '__main__':
    df = pd.read_csv(r'E:\mmaction2\work_dirs\autism_center_post_qa_fine_tune\test.csv')
    label = df['y']
    predict = np.argmax(df[df.columns[1:]].to_numpy(), axis=1)

    labels = np.array(label)[:, np.newaxis]
    scores = df[df.columns[1:]].to_numpy()
    max_k_preds = np.argsort(scores, axis=1)[:, -3:][:, ::-1]
    match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
    topk_acc_score = match_array.sum() / match_array.shape[0]


    result = {s: (((predict == i) & (label == i)).sum() / (label == i).sum(), ((match_array == 1) & (label == i)).sum() / (label == i).sum()) for i, s in enumerate(REAL_DATA_MOVEMENTS[:-4])}
    for k, (v1, v2) in result.items():
        print(f'{k} & {int(v1 * 100)}\\% & {int(v2 * 100)}\\% \\\\')
    plot_conf_matrix(df, norm=True)
    plot_scores_heatmap(df)

    df2 = pd.read_csv(r'E:\mmaction2\work_dirs\autism_center_post_qa\test.csv')
    label2 = df2['y']
    predict2 = np.argmax(df2[df2.columns[1:]].to_numpy(), axis=1)

    for i, c in enumerate(REAL_DATA_MOVEMENTS[:-4]):
        ft_correct = np.round((np.mean((label == i) & (predict == i) & (predict2 != i)) * 100), 3)
        ft_mistake = np.round((np.mean((label == i) & (predict != i) & (predict2 == i)) * 100), 3)
        print(i, c, f'fine_tune: {ft_correct}%', f'scratch: {ft_mistake}%')
    print(np.sum((label == predict) & (label != predict2)) - np.sum((label != predict) & (label == predict2)))

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
