import os
from os import path as osp
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from matplotlib.patches import Patch
import numpy as np
# import torch.fft as fft
from scipy.fftpack import fft
import pandas as pd
import seaborn as sns
from sklearn import metrics
from tqdm import tqdm

from skeleton_tools.openpose_layouts.body import COCO_LAYOUT
from skeleton_tools.skeleton_visualization.numpy_visualizer import MMPoseVisualizer
from skeleton_tools.utils.constants import REAL_DATA_MOVEMENTS, NET_NAME, STEP_SIZE, LENGTH
from skeleton_tools.utils.evaluation_utils import intersect
from skeleton_tools.utils.tools import read_pkl, get_video_properties, init_directories, collect_labels

pd.set_option('display.expand_frame_repr', False)
sns.set_theme()

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
    fig.savefig(osp.join(dst_path, f'{filename}.png'), dpi=100)
    # plt.show()


def bar_plot_lenghts(ax, df):
    # fig, ax = plt.subplots()
    df['length'] = df['end_time'] - df['start_time'] - 2
    nactions = df.groupby('movement')['video'].count().sort_values()
    classes = nactions.index
    lengths = [df[df['movement'] == c]['length'].to_numpy() for c in classes]
    groups = [(0, 3), (3, 5), (5, 8), (8, np.inf)]
    # cmap = sns.color_palette("mako", as_cmap=True)
    prev = np.zeros(len(classes))
    # n = 255 // len(groups)
    cmap = ['354F52', '52796F', '84A98C', 'CAD2C5']

    for i, (min_len, max_len) in enumerate(groups):
        counts = [np.count_nonzero(l[(min_len < l) & (l <= max_len)]) for l in lengths]
        # ax.barh(classes, counts, color=cmap(i * n), left=prev)
        ax.barh(classes, counts, color=f'#{cmap[i]}', left=prev)
        prev += counts
    legend = [f'{min_len}s - {max_len}s' for (min_len, max_len) in groups]
    legend[-1] = f'{groups[-1][0]}s $\leq$'
    ax.legend(legend, loc='lower right', ncol=2)
    ax.set_xlabel('Number of videos')
    ax.set_ylabel('Class')
    # plt.savefig(f'resources/figs/{name}_videos_lengths.png', bbox_inches='tight')
    # plt.show()


def bar_plot_unique_children(ax, df):
    # fig, ax = plt.subplots()
    df['ckey'] = df['video'].apply(lambda s: s.split('_')[0])
    gp = df.groupby(['movement'])['ckey'].nunique().sort_values()
    ax.barh(gp.index, gp.values / df['ckey'].nunique() * 100, color='#354F52')
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        ax.annotate(f'{int(width)}%', (x + width + 4, y + height * 0.25), ha='center')
    ax.set_xlim(0, 100)
    ax.set_xticks(ticks=np.arange(0, 101, 20), labels=[f'{i}%' for i in np.arange(0, 101, 20)])
    ax.set_xlabel('Percentage of unique children')
    # ax.set_ylabel('Class')
    # plt.savefig(f'resources/figs/{name}_unique_children.png', bbox_inches='tight')
    # plt.show()

# def bar_plot_actions_count_dist(df, name):
#     fig, ax = plt.subplots()
#     df['assessment'] = df['video'].apply(lambda s: '_'.join(s.split('_')[:-2]))
#     gp = df.groupby('assessment')['video'].count()
#     _df = pd.DataFrame(columns=['video', 'actions_count'], data=np.array([gp.index, gp.values]).T)
#     ax = sns.histplot(data=_df, x="actions_count", bins=max(_df.shape[0] // 10, 20), kde=True)
#     ax.set(title=name, xlabel='Actions count', ylabel='Assessments count')
#     plt.savefig(f'resources/figs/{name}_actions_count_dist.png', bbox_inches='tight')
#     plt.show()

def bar_plot_actions_count_dist(ax, dfs, names):
    colors = [f'#{i}' for i in ['6A4C93', '1982C4', '8AC926', 'FF595E']]
    _dfs = []
    for df, name in zip(dfs, names):
        df['assessment'] = df['video'].apply(lambda s: '_'.join(s.split('_')[:-2]))
        gp = df.groupby('assessment')['video'].count()
        _dfs.append(pd.DataFrame(columns=['video', 'actions_count', 'legend'], data=np.array([gp.index, gp.values, [name] * len(gp)]).T))
    for color, _df in zip(colors, _dfs):
        sns.distplot(_df['actions_count'], hist=False, ax=ax, color=color) # TODO: kde_kws={'bw':0.25} , check qq plot with ilan
    for l in ax.lines:
        x1 = l.get_xydata()[:, 0]
        y1 = l.get_xydata()[:, 1]
        c = l.get_color()
        ax.fill_between(x1, y1, alpha=0.1, color=c)
    ax.set_xlim(right=200)
    # _df = pd.concat(_dfs).reset_index(drop=True)
    # ax = sns.displot(data=_df, x="actions_count", hue='legend', multiple='stack', kind='kde')
    ax.set(xlabel='Actions count')
    ax.legend(labels=names, borderaxespad=2)
    # plt.savefig(f'resources/figs/actions_count_dist.png', bbox_inches='tight')
    # plt.show()

def bar_plot_actions_length_dist(ax, dfs, names):
    colors=[f'#{i}' for i in ['6A4C93', '1982C4', '8AC926', 'FF595E']]
    _dfs = []
    for df, name in zip(dfs, names):
        df['length'] = df['end_time'] - df['start_time']
        df['assessment'] = df['video'].apply(lambda s: '_'.join(s.split('_')[:-2]))
        gp = df.groupby('assessment')['length'].mean()
        _dfs.append(pd.DataFrame(columns=['video', 'mean_length', 'legend'], data=np.array([gp.index, gp.values, [name] * len(gp)]).T))
    for color, _df in zip(colors, _dfs):
        sns.distplot(_df['mean_length'], hist=False, ax=ax, color=color)
    for l in ax.lines:
        x1 = l.get_xydata()[:, 0]
        y1 = l.get_xydata()[:, 1]
        c = l.get_color()
        ax.fill_between(x1, y1, alpha=0.1, color=c)
    ax.set_xlim(right=50)
    # _df = pd.concat(_dfs).reset_index(drop=True)
    # ax = sns.displot(data=_df, x="mean_length", hue='legend', multiple='stack', kind='kde')
    ax.set(xlabel='Mean length')
    ax.legend(labels=names, borderaxespad=2)
    # plt.savefig(f'resources/figs/mean_length_dist.png', bbox_inches='tight')
    # plt.show()


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

def get_intersection(interval1, interval2):
    new_min = max(interval1[0], interval2[0])
    new_max = min(interval1[1], interval2[1])
    return [new_min, new_max] if new_min <= new_max else None

def draw_net_confidence(ax, jordi, agg, humans):
    jordi['window'] = (jordi['start_frame'] + jordi['end_frame']) / 2
    sns.lineplot(data=jordi, x='window', y='stereotypical_score', color='#264653', ax=ax)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_xlim(xmin=0, xmax=jordi['end_frame'].max() - jordi['end_frame'].max() % -1000)
    ax.set_xlabel("")
    ax.set_ylabel("")
    for i, row in agg.iterrows():
        ax.axvspan(row['start_frame'], row['end_frame'], alpha=.5, color='#FFBE0B')
    for i, row in humans.iterrows():
        ax.axvspan(row['start_frame'], row['end_frame'], alpha=.5, color='#3A86FF')

    jordi_intervals = agg[['start_frame', 'end_frame']].values.tolist()
    human_intervals = humans[['start_frame', 'end_frame']].values.tolist()
    intersection = [x for x in (get_intersection(y, z) for y in jordi_intervals for z in human_intervals) if x is not None]
    intersection_length = sum([(t - s) for s, t in intersection])
    union_length = sum([(t-s) for s,t in (jordi_intervals + human_intervals)]) - intersection_length
    ax.text(0.005, 0.92, f'IoU: {round(intersection_length / union_length * 100)}%', fontsize=25, ha='left', va='center', transform=ax.transAxes)


def aggregate_b(df): # TODO: Decide if using
    _df = pd.DataFrame(columns=df.columns)
    for i in range(0, df['end_frame'].max(), 30):
        sdf = df[(df['start_frame'] <= i) & (i < df['end_frame'])]
        _df.loc[_df.shape[0]] = [df['video'].loc[0], -1, -1, i, i + 30, -1, pd.to_datetime('now'), 'JORDI', sdf['stereotypical_score'].mean()]
    return _df

def draw_confidence_for_assessment(root, files, human_labels_path, show=False):
    init_directories(osp.join(root, 'figs'))
    assessment = ' '.join(files[0].split('_')[:-2])
    fig, axs = plt.subplots(len(files), figsize=(100, 20))
    fig.text(0.513, 0.98, r'$\bf{Model\ score\ for\ assessment:}$' + assessment, ha='center', va='top', size=60)
    fig.text(0.513, 0.9, f'Score threshold: {0.8}', ha='center', va='top', size=40)
    fig.text(0.513, 0.06, 'Frame', ha='center', size=45)
    fig.text(0.002, 0.5, 'Score', va='center', rotation='vertical', size=45)
    legend_elements = [Patch(facecolor='#FFBE0B', label='Jordi'),
                       Patch(facecolor='#3A86FF', label='Human')]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.95), fancybox=True, framealpha=0.5, fontsize=40)
    humans = pd.read_csv(human_labels_path)
    humans['basename'] = humans['video'].apply(lambda v: osp.splitext(v)[0])
    for ax, file in zip(axs, files):
        camid = file.split('_')[-1][0]
        name = osp.splitext(file)[0]
        jordi, agg = pd.read_csv(osp.join(root, name, 'jordi', f'{name}_scores.csv')), \
                     pd.read_csv(osp.join(root, name, 'jordi', f'{name}_annotations.csv'))
        agg = agg[(agg['movement'] == 1) | (agg['movement'] == 'Stereotypical')]
        draw_net_confidence(ax,
                            jordi,
                            agg,
                            humans[humans['basename'] == file])
        ax.set_ylabel(f'Camera {camid}', size=25)
    fig.tight_layout(rect=[0.01, 0.1, 0.99, 0.85])
    if show:
        fig.show()
    fig.savefig(osp.join(root, 'figs', f'{assessment}.png'))


def export_frames_for_figure():
    jordi = ['22210917', '666325510']
    files = [f for f in os.listdir(r'D:\data\lancet_submission\train\segmented_videos') if f.split('_')[0] in jordi]

    root = r'D:\data\lancet_submission\train'
    # files = ['100670545_Linguistic_Clinical_090720_0909_2_Spinning in circle_13543_14028.avi',
    #          '101614003_338720063_ADOS_191217_1244_1_Spinning in circle_24660_24810.avi',
    #          '101615086_ADOS_Clinical_030320_1301_2_Head movement_30329_30480.avi',
    #          '101631802_ADOS_Clinical_050120_1107_3_Body rocking_16633_17179.avi',
    #          '101631802_ADOS_Clinical_050120_1107_4_Body rocking_16633_17179.avi',
    #          '101857828_339532814_ADOS_110618_1216_1_Hand flapping_31110_31260.avi',
    #          '101991079_ADOS_Clinical_020220_0926_4_Head movement_3211_3332.avi',
    #          '102033871_ADOS_Clinical_191119_1216_1_Hand flapping_89295_89447.avi',
    #          '102033871_Cognitive_Clinical_041219_1301_4_Hand flapping_84776_84928.avi',
    #          '102105601_340358720_ADOS_070118_0932_1_Hand flapping_51780_52320.avi']
    v = MMPoseVisualizer(graph_layout=COCO_LAYOUT, blur_face=True)
    for f in files:
        name, ext = osp.splitext(f)
        j = read_pkl(osp.join(root, 'skeletons', 'raw', f'{name}.pkl'))
        vid = osp.join(root, 'segmented_videos', f)
        v.create_double_frame_video(vid, j, f'C:/Users/owner/Desktop/out/vid_{f}')
        v.export_frames(vid, j, f'C:/Users/owner/Desktop/out/{name}')


if __name__ == '__main__':
    train = pd.read_csv(r'Z:\Users\TalBarami\lancet_submission_data\annotations\labels.csv')
    pre_qa = pd.read_csv(r'Z:\Users\TalBarami\JORDI_50_vids_benchmark\annotations\human_pre_qa.csv')
    post_qa = pd.read_csv(r'Z:\Users\TalBarami\JORDI_50_vids_benchmark\annotations\human_post_qa.csv')
    post_qa['assessment'] = post_qa['video'].apply(lambda v: '_'.join(v.split('_')[:-2]))
    jordi = collect_labels(r'Z:\Users\TalBarami\JORDI_50_vids_benchmark\JORDIv4', 'jordi/binary_cd_epoch_37.pth')
    jordi = jordi[jordi['assessment'].isin(post_qa['assessment'].unique())]
    jordi = jordi[jordi['movement'] != 'NoAction']
    pre_qa, post_qa, jordi = intersect(pre_qa, post_qa, jordi, on='video')

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 6)
    bar_plot_lenghts(ax1, train)
    bar_plot_unique_children(ax2, train)
    fig.tight_layout()
    plt.savefig(f'resources/figs/train_statistics.png', bbox_inches='tight')
    plt.show()
    # exit()
    # export_frames_for_figure()
    # exit()
    sns.set_style(style='white')
    dfs, names = zip(*[(train, 'Train'), (pre_qa, 'Pre QA'), (post_qa, 'Post QA'), (jordi, 'Model')])
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    bar_plot_actions_count_dist(ax, dfs, names)
    fig.tight_layout()
    plt.savefig(f'resources/figs/actions_count_dist.png', bbox_inches='tight')
    plt.show()
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    bar_plot_actions_length_dist(ax, dfs, names)
    fig.tight_layout()
    plt.savefig(f'resources/figs/mean_length_dist.png', bbox_inches='tight')
    plt.show()

    # df = pd.read_csv(r'E:\mmaction2\work_dirs\autism_center_post_qa_fine_tune\test.csv')
    # label = df['y']
    # predict = np.argmax(df[df.columns[1:]].to_numpy(), axis=1)
    #
    # labels = np.array(label)[:, np.newaxis]
    # scores = df[df.columns[1:]].to_numpy()
    # max_k_preds = np.argsort(scores, axis=1)[:, -3:][:, ::-1]
    # match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
    # topk_acc_score = match_array.sum() / match_array.shape[0]
    #
    # result = {s: (((predict == i) & (label == i)).sum() / (label == i).sum(), ((match_array == 1) & (label == i)).sum() / (label == i).sum()) for i, s in enumerate(REAL_DATA_MOVEMENTS[:-4])}
    # for k, (v1, v2) in result.items():
    #     print(f'{k} & {int(v1 * 100)}\\% & {int(v2 * 100)}\\% \\\\')
    # plot_conf_matrix(df, norm=True)
    # plot_scores_heatmap(df)
    #
    # df2 = pd.read_csv(r'E:\mmaction2\work_dirs\autism_center_post_qa\test.csv')
    # label2 = df2['y']
    # predict2 = np.argmax(df2[df2.columns[1:]].to_numpy(), axis=1)
    #
    # for i, c in enumerate(REAL_DATA_MOVEMENTS[:-4]):
    #     ft_correct = np.round((np.mean((label == i) & (predict == i) & (predict2 != i)) * 100), 3)
    #     ft_mistake = np.round((np.mean((label == i) & (predict != i) & (predict2 == i)) * 100), 3)
    #     print(i, c, f'fine_tune: {ft_correct}%', f'scratch: {ft_mistake}%')
    # print(np.sum((label == predict) & (label != predict2)) - np.sum((label != predict) & (label == predict2)))

    # vinfos = []
    # for root, dirs, files in os.walk(r'D:\datasets\tagging_hadas&dor\raw_videos'):
    #     for f in files:
    #         vinfos.append(get_video_properties(osp.join(root, f)))
    # resolution = set([v[0] for v in vinfos])
    # fps = np.unique([v[1] for v in vinfos])
    # df_path = r'C:\Users\owner\PycharmProjects\RepetitiveMotionRecognition\resources\labels_complete_updated.csv'
    # bar_plot_lenghts(df_path)
    # bar_plot_unique_children(df_path)
    # skeletons_root = r'D:\datasets\autism_center\skeletons_filtered\data'
    # videos_root = r'D:\datasets\autism_center\segmented_videos'
    #
    # files = [osp.splitext(f)[0] for f in os.listdir(videos_root) if f.endswith('.avi')]
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
    #         data = read_json(osp.join(skeletons_root, f'{filename}.json'))['data']
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
