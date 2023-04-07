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
from scipy import stats

from skeleton_tools.openpose_layouts.body import COCO_LAYOUT
from skeleton_tools.skeleton_visualization.mmpose_visualizer import MMPoseVisualizer
from skeleton_tools.utils.constants import REAL_DATA_MOVEMENTS, NET_NAME, STEP_SIZE, LENGTH
from skeleton_tools.utils.evaluation_utils import intersect, collect_predictions, prepare
from skeleton_tools.utils.tools import read_pkl, get_video_properties, init_directories, scan_db

pd.set_option('display.expand_frame_repr', False)
sns.set_theme()
sns.set(font_scale=1.2)

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
    fig.savefig(osp.join(dst_path, f'{filename}.png'), dpi=240)
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
    cmap = ['181818', '696969', '989898', 'BEBEBE']

    for i, (min_len, max_len) in enumerate(groups):
        counts = [np.count_nonzero(l[(min_len < l) & (l <= max_len)]) for l in lengths]
        # ax.barh(classes, counts, color=cmap(i * n), left=prev)
        ax.barh(classes, counts, color=f'#{cmap[i]}', left=prev)
        prev += counts
    legend = [f'{min_len}s - {max_len}s' for (min_len, max_len) in groups]
    legend[-1] = f'{groups[-1][0]}s $\leq$'
    ax.legend(legend, loc='lower right', ncol=2)
    ax.set_xlabel('Number of video segments')
    # plt.savefig(f'resources/figs/{name}_videos_lengths.png', bbox_inches='tight')
    # plt.show()


def bar_plot_unique_children(ax, df):
    # fig, ax = plt.subplots()
    df['ckey'] = df['video'].apply(lambda s: s.split('_')[0])
    gp = df.groupby(['movement'])['ckey'].nunique().sort_values()
    ax.barh(gp.index, gp.values / df['ckey'].nunique() * 100, color='#696969')
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        ax.annotate(f'{int(width)}%', (x + width + 5.5, y + height * 0.1), ha='center')
    ax.set_xlim(0, 100)
    ax.set_xticks(ticks=np.arange(0, 101, 20), labels=[f'{i}%' for i in np.arange(0, 101, 20)])
    ax.set_xlabel('Percentage of children')

def bar_plot_actions_count_dist(ax, df):
    gp = df.groupby(['assessment', 'legend']).agg({'video': 'count'}).reset_index()
    gp = gp[(np.abs(stats.zscore(gp['video'])) < 2)]
    sns.histplot(data=gp, x='video', hue='legend', kde=True, ax=ax)
    # sns.displot(data=gp, x='video', hue='legend', kde=True, ax=ax)
    ax.set(xlabel='Actions count', ylabel='Assessments count')

def bar_plot_actions_length_dist(ax, df):
    gp = df.groupby(['assessment', 'legend']).agg({'length': 'sum', 'length_seconds': 'first'}).reset_index()
    gp = gp[(np.abs(stats.zscore(gp['length'])) < 2)]
    gp['rel_length'] = gp['length'] / gp['length_seconds']
    # sns.histplot(data=gp, x='rel_length', hue='legend', multiple='dodge')
    sns.histplot(data=gp, x='rel_length', hue='legend', kde=True, ax=ax)
    # sns.displot(data=gp, x='rel_length', hue='legend', kde=True, ax=ax)
    ax.set(xlabel='Stereotypical relative length', ylabel='Assessments count')

def plot_model_vs_human_actions_count(ax, df1, df2):
    g1 = df1.groupby('assessment').agg({'name': 'count'}).reset_index()
    g2 = df2.groupby('assessment').agg({'name': 'count'}).reset_index()
    df = pd.merge(g1, g2, on='assessment', how='inner')
    df.columns = ['assessment', 'human_count', 'model_count']
    # df = df[(np.abs(stats.zscore(df['human_count'])) < 2) & (np.abs(stats.zscore(df['model_count'])) < 2)]
    m, n = df['human_count'].max(), df['model_count'].max()
    k = min(m, n)
    sns.scatterplot(data=df, x='human_count', y='model_count', ax=ax)
    sns.regplot(data=df, x='human_count', y='model_count', ax=ax)
    ax.plot((0, k), (0, k))
    ax.set(xlabel='Human actions count', ylabel='Model actions count', xlim=(0, m), ylim=(0, n))

def plot_model_vs_human_rel_length(ax, df1, df2):
    gp = []
    for df in [df1, df2]:
        g = df.groupby('assessment').agg({'length': 'sum', 'length_seconds': 'first'}).reset_index()
        g['rel_length'] = g['length'] / g['length_seconds']
        gp.append(g[['assessment', 'rel_length']])
    df = pd.merge(gp[0], gp[1], on='assessment', how='inner')
    df.columns = ['assessment', 'human_rel_length', 'model_rel_length']
    # df = df[(np.abs(stats.zscore(df['human_rel_length'])) < 2) & (np.abs(stats.zscore(df['model_rel_length'])) < 2)]
    m, n = df['human_rel_length'].max(), df['model_rel_length'].max()
    k = min(m, n)
    sns.scatterplot(data=df, x='human_rel_length', y='model_rel_length', ax=ax)
    sns.regplot(data=df, x='human_rel_length', y='model_rel_length', ax=ax)
    ax.plot((0, k), (0, k))
    ax.set(xlabel='Stereotypical relative length (human)', ylabel='Stereotypical relative length (model)', xlim=(0, m), ylim=(0, m))

def bland_altman(ax, df1, df2):
    g1 = df1.groupby('assessment').agg({'name': 'count'}).reset_index()
    g2 = df2.groupby('assessment').agg({'name': 'count'}).reset_index()
    df = pd.merge(g1, g2, on='assessment', how='inner')
    df.columns = ['assessment', 'human_count', 'model_count']
    df = df[(np.abs(stats.zscore(df['human_count'])) < 2) & (np.abs(stats.zscore(df['model_count'])) < 2)]
    df['count_diff'] = df['human_count'] - df['model_count']
    # sns.scatterplot(data=df, x='human_count', y='count_diff', ax=ax)
    m = df['count_diff'].mean()
    s = df['count_diff'].std()

    ax.scatter(np.mean([df['human_count'], df['model_count']], axis=0), df['count_diff'])
    ax.axhline(m, color='gray', linestyle='-')
    ax.axhline(0, color='green', linestyle='-')
    ax.axhline(m + 1.96 * s, color='gray', linestyle='--')
    ax.axhline(m - 1.96 * s, color='gray', linestyle='--')
    ax.set(xlabel='Mean of human and model actions count', ylabel='Difference between human and model actions count', ylim=(-4 * s, 4 * s))

def histogram_count(ax, df):
    _gp = df.groupby(['assessment', 'video']).agg({'start_time': 'count'}).groupby('assessment').agg({'start_time': 'max'}).reset_index()
    gp = _gp[(np.abs(stats.zscore(_gp['start_time'])) < 3)]
    sns.histplot(data=gp, x='start_time', ax=ax)
    ax.set(xlabel='Actions count', ylabel='Assessments count')

def histogram_relative_length(ax, df):
    df['relative_length'] = (df['end_frame'] - df['start_frame']) / df['frame_count']
    _gp = df.groupby(['assessment', 'video']).agg({'relative_length': 'sum'}).groupby('assessment').agg({'relative_length': 'max'}).reset_index()
    gp = _gp[(np.abs(stats.zscore(_gp['relative_length'])) < 3)]
    sns.histplot(data=gp, x='relative_length', ax=ax)
    ax.set(xlabel='Stereotypical relative length', ylabel='Assessments count')

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


def display(f, size=(8, 6), show=False, save=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(*size)
    f(ax)
    fig.tight_layout()
    if save:
        fig.savefig(f'resources/figs/{save}.png', dpi=240)
    if show:
        plt.show()
    return fig

def model_statistics(dfs, names):
    db = scan_db()
    # train = pd.read_csv(r'Z:\Users\TalBarami\lancet_submission_data\annotations\labels.csv')
    # # pre_qa = pd.read_csv(r'Z:\Users\TalBarami\JORDI_50_vids_benchmark\annotations\human_pre_qa.csv')
    # post_qa = pd.read_csv(r'Z:\Users\TalBarami\JORDI_50_vids_benchmark\annotations\human_post_qa2.csv')
    # jordi = collect_labels(r'Z:\Users\TalBarami\JORDI_50_vids_benchmark\JORDIv4', 'jordi/binary_cd_epoch_37.pth')
    # jordi = jordi[jordi['movement'] != 'NoAction']
    # exclude = ['666852718_ADOS_Clinical_301120', '663954442_ADOS_Clinical_210920', '666814726_ADOS_Clinical_110717']
    # pre_qa, post_qa, jordi = intersect(pre_qa, post_qa, jordi, on='video')
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.set_size_inches(15, 6)
    # bar_plot_lenghts(ax1, train)
    # bar_plot_unique_children(ax2, train)
    # fig.tight_layout()
    # plt.savefig(f'resources/figs/train_statistics.png', bbox_inches='tight')
    # plt.show()

    # dfs, names = zip(*[(post_qa, 'Human'), (jordi, 'Model')])
    legend = {'Human': 'Human', 'cv0.pth': 'Model'}
    for df, name in zip(dfs, names):
        df['basename'] = df['video'].apply(lambda v: osp.splitext(v)[0])
        df['assessment'] = df['video'].apply(lambda v: '_'.join(v.split('_')[:-2]))
        df['name'] = name
        # df.drop(df[df['video'].isin(exclude)].index, inplace=True)
    df = pd.concat(dfs)
    df = df[df['movement'] != 'NoAction']
    df['assessment'] = df['video'].apply(lambda v: '_'.join(v.split('_')[:-1]))
    df['child'] = df['assessment'].apply(lambda a: a.split('_')[0])
    df['length'] = df['end_time'] - df['start_time']
    df['legend'] = df['name'].apply(lambda n: legend[n])

    # df[['path', 'width', 'height', 'fps', 'frame_count', 'length_seconds']] = \
    #     df.apply(lambda v: db[db['video'] == v['video']].iloc[0][['path', 'width', 'height', 'fps', 'frame_count', 'length_seconds']],
    #              axis=1,
    #              result_type="expand")
    display(lambda ax: bar_plot_actions_count_dist(ax, df), show=True, save='actions_count_dist', size=(8, 8))
    display(lambda ax: bar_plot_actions_length_dist(ax, df), show=True, save='actions_length_dist', size=(8, 8))
    display(lambda ax: plot_model_vs_human_actions_count(ax, df[df['name'] == 'Human'], df[df['name'] != 'Human']), show=True, save='model_vs_human_actions_count')
    display(lambda ax: plot_model_vs_human_rel_length(ax, df[df['name'] == 'Human'], df[df['name'] != 'Human']), show=True, save='model_vs_human_rel_length')
    display(lambda ax: bland_altman(ax, df[df['name'] == 'Human'], df[df['name'] != 'Human']), show=True, save='bland_altman')


def data_statistics(_df):
    _df.loc[_df[_df['movement'].isna()].index, 'movement'] = 'Other'
    df = pd.DataFrame(columns=_df.columns)
    for i, row in _df.iterrows():
        movements = row['movement'].split(',')
        for m in movements:
            rc = row.copy()
            rc['movement'] = m
            df.loc[df.shape[0]] = rc
    display(lambda ax: bar_plot_lenghts(ax, df), show=True, save='data_stats')
    display(lambda ax: bar_plot_unique_children(ax, df), show=True, save='children_stats')
    display(lambda ax: histogram_count(ax, df), show=True, save='count_stats')
    display(lambda ax: histogram_relative_length(ax, df), show=True, save='relative_length_stats')


if __name__ == '__main__':
    root = r'Z:\Users\TalBarami\jordi_cross_validation'
    sns.set_style(style='white')
    human = prepare(pd.read_csv(r'Z:\Users\TalBarami\lancet_submission_data\annotations\combined.csv'))
    # data_statistics(human)

    model = 'cv0.pth'
    df, summary_df, agg_df, summary_agg_df = collect_predictions(root, model_name=model)
    model_statistics([df[df['annotator'] != NET_NAME], df[df['annotator'] == NET_NAME]], ['Human', model])

    # model = 'cv1.pth'
    # df, summary_df, agg_df, summary_agg_df = collect_predictions(root, model_name=model)
    # generate_statistics([df[df['annotator'] != NET_NAME], df[df['annotator'] == NET_NAME]], ['Human', model])
    #
    # model = 'cv2.pth'
    # df, summary_df, agg_df, summary_agg_df = collect_predictions(r'\\ac-s1\Data\Autism Center\Users\TalBarami\JORDI_50_vids_benchmark\JORDIv4', model_name='binary_cd_epoch_37.pth')
    # generate_statistics([df[df['annotator'] != NET_NAME], df[df['annotator'] == NET_NAME]], ['Human', 'binary_cd_epoch_37.pth'])

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
