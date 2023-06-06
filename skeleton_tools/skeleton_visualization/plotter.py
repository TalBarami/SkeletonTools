import os
from os import path as osp

import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.ticker as mtick
from scipy import stats
# import torch.fft as fft
from scipy.fftpack import fft
from skeleton_tools.openpose_layouts.body import COCO_LAYOUT, BODY_25_LAYOUT
from skeleton_tools.skeleton_visualization.data_prepare.data_extract import MMPoseDataExtractor
from skeleton_tools.skeleton_visualization.paint_components.frame_painters.base_painters import GlobalPainter, BlurPainter
from skeleton_tools.skeleton_visualization.paint_components.frame_painters.local_painters import GraphPainter
from skeleton_tools.skeleton_visualization.visualizer import VideoCreator
from skeleton_tools.utils.constants import NET_NAME, DB_PATH
from skeleton_tools.utils.evaluation_utils import intersect, collect_predictions, prepare
from skeleton_tools.utils.tools import init_directories

pd.set_option('display.expand_frame_repr', False)
sns.set_theme()
sns.set(font_scale=1.4)


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
    gp = df.groupby(['assessment', 'video', 'Legend']).agg({'movement': 'count', 'length_seconds': 'first'}).reset_index()
    gp = gp.groupby(['assessment', 'Legend']).agg({'movement': 'max', 'length_seconds': 'max'}).reset_index()
    gp['count_per_minute'] = gp['movement'] / gp['length_seconds'] * 60
    # gp = gp[(np.abs(stats.zscore(gp['count_per_minute'])) < 2)]
    n = 32
    sns.histplot(data=gp, x='count_per_minute', hue='Legend', bins=n, kde=True, ax=ax)
    ax.set(xlabel='SMMs per minute', ylabel='Assessments count')
    ax.legend(['Distribution'])


def bar_plot_actions_length_dist(ax, df):
    gp = df.groupby(['assessment', 'video', 'Legend']).agg({'length': 'sum', 'length_seconds': 'first'}).reset_index()
    gp = gp.groupby(['assessment', 'Legend']).agg({'length': 'max', 'length_seconds': 'max'}).reset_index()
    gp['rel_length'] = gp['length'] / gp['length_seconds'] * 100
    # gp = gp[(np.abs(stats.zscore(gp['rel_length'])) < 2)]
    n = 32
    sns.histplot(data=gp, x='rel_length', hue='Legend', bins=n, kde=True, ax=ax)
    ax.set(xlabel='Percentage of time with SMMs', ylabel='Assessments count')
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(['Distribution'])


def plot_model_vs_human_actions_count(ax, df1, df2):
    g1 = df1.groupby('assessment').agg({'name': 'count'}).reset_index()
    g2 = df2.groupby('assessment').agg({'name': 'count'}).reset_index()
    df = pd.merge(g1, g2, on='assessment', how='inner')
    df.columns = ['assessment', 'human_count', 'model_count']
    df = df[(np.abs(stats.zscore(df['human_count'])) < 2) & (np.abs(stats.zscore(df['model_count'])) < 2)]
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
    df = df[(np.abs(stats.zscore(df['human_rel_length'])) < 2) & (np.abs(stats.zscore(df['model_rel_length'])) < 2)]
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
    union_length = sum([(t - s) for s, t in (jordi_intervals + human_intervals)]) - intersection_length
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
    ids = [666325510, 1018226485, 704767285]
    exclude = ['666325510_ADOS_Clinical_120320_0938_1', '666325510_ADOS_Clinical_120320_0938_2',
               '666325510_Cognitive_Clinical_120320_1056_1', '666325510_Cognitive_Clinical_120320_1057_2',
               '666325510_Cognitive_Clinical_170320_1037_1', '666325510_Cognitive_Clinical_170320_1038_2',
               '666325510_PLS_Clinical_170320_1250_1']
    # df = pd.read_csv(r'Z:\Users\TalBarami\lancet_submission_data\annotations\qa_230323_namefix.csv')
    df = pd.read_csv(r'Z:\Users\TalBarami\models_outputs\704767285_Cognitive_Control_300522_0848_1\jordi\cv0.pth\704767285_Cognitive_Control_300522_0848_1_annotations.csv')
    df = df[df['movement'] != 'NoAction']
    # df = df[df['child_id'].isin(ids)]
    # df = df[~df['video'].isin(exclude)]
    db = scan_db()

    out = r'D:\repos\SkeletonTools\resources\figs\sample_frames'
    # skeletons_dir = r'Z:\Users\TalBarami\jordi_cross_validation'
    skeletons_dir = r'Z:\Users\TalBarami\models_outputs'
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
    extractor = MMPoseDataExtractor(COCO_LAYOUT)
    v2 = VideoCreator(global_painters=[GlobalPainter(GraphPainter(COCO_LAYOUT, color=(150, 150, 0)))])
    v3 = VideoCreator(global_painters=[GlobalPainter(GraphPainter(COCO_LAYOUT))])
    for g, rows in df.groupby('video'):
        out_dir = osp.join(out, g)
        video_path = db[db['basename'] == g]['file_path'].iloc[0]
        skeleton_path = osp.join(skeletons_dir, g, 'jordi', f'{g}.pkl')
        skeleton_data = extractor(skeleton_path)
        v1 = VideoCreator([BlurPainter(skeleton_data)])
        init_directories(out_dir)
        for i, row in rows.iterrows():
            start, end = row['start_frame'], row['end_frame']
            # if not osp.exists(osp.join(out_dir, 'frames', f'{start}.jpg')):
            #     v1.create_image(video_path=video_path, data=skeleton_data, out_path=osp.join(out_dir, 'frames'), start=start, end=end)
            v1.create_video(video_path=video_path, data=skeleton_data, out_path=osp.join(out_dir, 'frames.avi'), start=start, end=end)
            # if not osp.exists(osp.join(out_dir, 'skeleton', f'{start}.jpg')):
            #     v2.create_image(data=skeleton_data, out_path=osp.join(out_dir, 'skeleton'), start=start, end=end)
            v2.create_video(data=skeleton_data, out_path=osp.join(out_dir, 'skeleton.avi'), start=start, end=end)
            # if not osp.exists(osp.join(out_dir, 'child_detect', f'{start}.jpg')):
            #     v3.create_image(data=skeleton_data, out_path=osp.join(out_dir, 'child_detect'), start=start, end=end)
            v3.create_video(data=skeleton_data, out_path=osp.join(out_dir, 'child_detect.avi'), start=start, end=end)


def take_specific_frames():
    _root = r'D:\repos\SkeletonTools\resources\figs\sample_frames'
    data = [('666325510_PLS_Clinical_170320_1251_4', '30'),
            ('666325510_Cognitive_Clinical_120320_1057_4', '88563'),
            ('704767285_Cognitive_Control_300522_0848_1', '30810.0')]
    file, frame_str = data[2]
    frame_num = int(float(frame_str))
    root = osp.join(_root, file)

    f1 = f'frames_{frame_str}.avi'
    f2 = f'skeleton_{frame_str}.avi'
    f3 = f'child_detect_{frame_str}.avi'
    out = osp.join(root, 'figure')
    init_directories(out)

    for f in [f1, f2, f3]:
        path = osp.join(root, f)
        type = f.split('_')[0]
        cap = cv2.VideoCapture(path)
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # if i % 10 == 0:
                cv2.imwrite(osp.join(out, f'{type}_{file}_{frame_num + i}.jpg'), frame)
                i += 1
            else:
                break
        cap.release()


def concatenate(outputs_dir, name):
    frames_dir = osp.join(outputs_dir, name)
    files = os.listdir(frames_dir)
    frames = [f for f in files if 'frames' in f]
    skeletons = [f for f in files if 'skeleton' in f]
    child_detect = [f for f in files if 'child' in f]
    for files in [frames, skeletons, child_detect]:
        files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
        imgs = [cv2.imread(osp.join(frames_dir, f)) for f in files]
        imgs = [i[700:1500, 300:900] for i in imgs]
        img = np.concatenate(imgs, axis=1)
        cv2.imwrite(osp.join(outputs_dir, f'{files[0]}.jpg'), img)


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
    dfs = intersect(*dfs)
    # train = pd.read_csv(r'Z:\Users\TalBarami\lancet_submission_data\annotations\labels.csv')
    # # pre_qa = pd.read_csv(r'Z:\Users\TalBarami\JORDI_50_vids_benchmark\annotations\human_pre_qa.csv')
    # post_qa = pd.read_csv(r'Z:\Users\TalBarami\JORDI_50_vids_benchmark\annotations\human_post_qa2.csv')
    # jordi = collect_labels(r'Z:\Users\TalBarami\JORDI_50_vids_benchmark\JORDIv4', 'jordi/binary_cd_epoch_37.pth')
    # jordi = jordi[jordi['movement'] != 'NoAction']
    exclude = ['666852718', '663954442', '666814726']
    # pre_qa, post_qa, jordi = intersect(pre_qa, post_qa, jordi, on='video')
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.set_size_inches(15, 6)
    # bar_plot_lenghts(ax1, train)
    # bar_plot_unique_children(ax2, train)
    # fig.tight_layout()
    # plt.savefig(f'resources/figs/train_statistics.png', bbox_inches='tight')
    # plt.show()

    # dfs, names = zip(*[(post_qa, 'Human'), (jordi, 'Model')])
    for df, name in zip(dfs, names):
        df['basename'] = df['video'].apply(lambda v: osp.splitext(v)[0])
        df['assessment'] = df['video'].apply(lambda v: '_'.join(v.split('_')[:-2]))
        df['name'] = name
        # df.drop(df[df['video'].isin(exclude)].index, inplace=True)
    df = pd.concat(dfs)
    df = df[df['movement'] != 'NoAction']
    db = pd.read_csv(DB_PATH)
    df['assessment'] = df['video'].apply(lambda v: '_'.join(v.split('_')[:-1]))
    df['child'] = df['assessment'].apply(lambda a: a.split('_')[0])
    df['length'] = df['end_time'] - df['start_time']
    df['Legend'] = df['name']
    df['length_seconds'] = df['video'].apply(lambda v: db[db['basename'] == v]['length_seconds'].iloc[0])

    # df[['path', 'width', 'height', 'fps', 'frame_count', 'length_seconds']] = \
    #     df.apply(lambda v: db[db['video'] == v['video']].iloc[0][['path', 'width', 'height', 'fps', 'frame_count', 'length_seconds']],
    #              axis=1,
    #              result_type="expand")
    display(lambda ax: bar_plot_actions_count_dist(ax, df), show=True, save='actions_count_dist', size=(6, 6))
    display(lambda ax: bar_plot_actions_length_dist(ax, df), show=True, save='actions_length_dist', size=(6, 6))
    # display(lambda ax: plot_model_vs_human_actions_count(ax, df[df['name'] == 'Human'], df[df['name'] != 'Human']), show=True, save='model_vs_human_actions_count')
    # display(lambda ax: plot_model_vs_human_rel_length(ax, df[df['name'] == 'Human'], df[df['name'] != 'Human']), show=True, save='model_vs_human_rel_length')
    # display(lambda ax: bland_altman(ax, df[df['name'] == 'Human'], df[df['name'] != 'Human']), show=True, save='bland_altman')


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

    fig, ax = plt.subplots()
    gp = df.groupby(['assessment', 'Legend']).agg({'video': 'count'}).reset_index()
    gp = gp[(np.abs(stats.zscore(gp['video'])) < 2)]
    n = 16
    sns.histplot(data=gp, x='video', hue='Legend', bins=n, kde=True, ax=ax)
    ax.set(xlabel='Actions count', ylabel='Assessments count')
    fig.tight_layout()
    plt.show()


cids = [1006723600, 1009730632, 1012020277, 1014362914, 1016034706, 1016119861, 1016169022, 1017665404, 1017666865, 1017854941, 1017991189, 1018091428, 1018093207, 1018171321, 1019520097, 1019524600, 1019534968, 1019737348, 1019928946, 1020229279, 1020232399, 1020232741, 1020280537, 1020356866, 1020840187, 1021074145, 1021119094, 672641044, 1021205218, 1021205716, 1021218634, 1021220041, 1021226305, 1021229647, 1021247194, 1021255855, 1021259545, 1021264195, 1021264696, 1021280263, 1021280386, 1021311823, 1021313080, 1021374739, 1021379188, 1021396366, 1021410529, 1021775038, 1021794247, 1021801294, 1022031991, 1023996280, 1024252834, 1026126388, 1026131164, 1026131461, 1026623464, 1026631978, 1026643636, 1026666508, 1026671704, 1029200926, 1032308617, 1032318493, 1032336166, 1032346012, 1032358114, 1032467836, 1032548029, 1032581863, 1032613978, 1032669376, 1032693706, 1032737035, 666325510, 663911920, 664010395, 664022104, 664191175, 664209490, 664257328, 664292125, 664323535,
        664325617, 664650412, 664905604, 664973815, 665978449, 666058111, 666059845, 666101551, 666147613, 666169936, 666170974, 666198229, 666238693, 666260098, 666271387, 666273793, 1016164216, 666299845, 666327940, 666398206, 666725203, 666728917, 666755029, 666763069, 666770197, 666783493, 666789355, 666808807, 666885463, 666896602, 669769576, 671332390, 671332663, 671496484, 671499553, 671609842, 672624811, 672644434, 672652330, 672652900, 672984133, 673002079, 673019275, 673020094, 673021180, 673039195, 673057642, 673058110, 673079620, 673086178, 673098976, 673109044, 673119394, 673140628, 673161793, 673165576, 673179490, 673179910, 673194088, 673243234, 673268731, 673273975, 673950985, 675344092, 675350128, 675556051, 675574717, 675582640, 675586522, 675605971, 675627832, 675667054, 675670297, 675698041, 675702529, 675734524, 675737197, 675746851, 675773878, 675794917, 675807640, 675818950, 675827389, 675835975, 675844315, 679538797, 681787636, 984664126, 991680802, 1017096055,
        1021784743, 1024402006, 1027718011, 1029179797, 1029183859, 1029690886, 1030823962, 1031466274, 1032314131, 1032341833, 1032372673, 1032377323, 1032382294, 1032406999, 1032442435, 1032443998, 1032464122, 1032470011, 1032482908, 1032515131, 1032527656, 1032533449, 1032551617, 1032551638, 1032618016, 1032620911, 1032646177, 1032654025, 1032660589, 1032702517, 1032735409, 1032766588, 1032776491, 1032781165, 1034680636, 1034998021, 666789877, 666830041, 666911641, 668067349, 668082499, 670619293, 670931986, 671257546, 671280265, 671338009, 671611000, 672652306, 673049497, 673098007, 673155703, 673235122, 673372111, 675527842, 675793378, 675824320, 675839716, 675853075, 675873676, 675902650, 677426581, 678147601, 678745981, 679022293, 679257913, 679266529, 680005483, 680527114, 681746749, 683771299, 683842552, 684133783, 684544366, 685926727,
        666795838, 666814726, 667997179, 668041255, 669770491, 673038484, 1009772854, 1016158174, 1020741259, 1028846038, 1032316954, 663875689, 664007080]
if __name__ == '__main__':
    # export_frames_for_figure()
    # take_specific_frames()
    # concatenate(r'D:\repos\SkeletonTools\resources\figs\sample_frames\704767285_Cognitive_Control_300522_0848_1\figure\out', 'a')
    # concatenate(r'D:\repos\SkeletonTools\resources\figs\sample_frames\704767285_Cognitive_Control_300522_0848_1\figure\out', 'b')
    # exit()
    root = r'Z:\Users\TalBarami\jordi_cross_validation'
    sns.set_style(style='white')
    # human = prepare(pd.read_csv(r'Z:\Users\TalBarami\lancet_submission_data\annotations\combined.csv'))
    # data_statistics(human)

    # model = 'cv0.pth'
    # df, summary_df, agg_df, summary_agg_df = collect_predictions(root, model_name='cv0.pth')
    # human, model = df[df['annotator'] != NET_NAME], df[df['annotator'] == NET_NAME]
    # df = pd.read_csv(r'Z:\Users\TalBarami\videos_qa\qa_processed.csv')
    # model = df.dropna(subset=['jordi_start', 'jordi_end']).copy()
    # human = df.dropna(subset=['human_start', 'human_end']).copy()
    # human['start_time'], human['end_time'], human['movement'] = human['human_start'] / model['fps'], human['human_end'] / human['fps'], human['qa_hadas']
    # model['start_time'], model['end_time'], model['movement'] = model['jordi_start'] / model['fps'], model['jordi_end'] / model['fps'], model['jordi_annotation']
    # model_statistics([human, model], ['Human', 'Model'])

    # Training data statistics:
    df = pd.read_csv(r'Z:\Users\TalBarami\videos_qa\qa_processed.csv')
    df = df[df['human_start'].notna()]
    df = df[df['human_annotation'] != 'NoAction']
    df = df[df['child_id'].isin(cids)]
    df['human_start'] = df['human_start'] / df['fps']
    df['human_end'] = df['human_end'] / df['fps']
    df[['start_time', 'end_time', 'movement']] = df[['human_start', 'human_end', 'human_annotation']]

    _df = df.copy()
    df = pd.DataFrame(columns=['child_key', 'assessment', 'start', 'end', 'fps', 'frame_count', 'length_seconds', 'valid_frames'])
    for i, row in _df.iterrows():
        child_key, assessment, start, end, fps, frame_count, length_seconds = row[['child_id', 'assessment', 'human_start', 'human_end', 'fps', 'frame_count', 'length_seconds']]
        if len(df[(df['child_key'] == child_key) & (df['assessment'] == assessment) & (df['start'] == start) & (df['end'] == end)]) > 0:
            continue
        df = df.append({'child_key': child_key, 'assessment': assessment, 'start': start, 'end': end, 'fps': fps, 'frame_count': frame_count, 'length_seconds': length_seconds, 'valid_frames': 0}, ignore_index=True)

    # TODO: Go over df, combine to info per assessment. Create new df.
    #  Take from skeletons data the number of visible-child frames per video. Calculate proportion of time & count
    #  Split train/test
    _db = pd.read_csv(r'Z:\Autism Center\recordings\redcap_db.csv')
    _db2 = pd.read_csv(r'Z:\Autism Center\recordings\db.csv')

    trn_df = pd.read_csv(r'Z:\Autism Center\Users\TalBarami\lancet_submission_data\train_children_stats.csv').dropna(subset='child_key')
    trn_cids = trn_df['child_key'].astype(int).unique()
    trn = _db[_db['child_key'].isin(trn_cids)]

    tst_df = pd.read_csv(r'Z:\Autism Center\Users\TalBarami\lancet_submission_data\test_children_stats.csv')
    tst_cids = tst_df['child_id'].unique()
    tst = _db[_db['child_key'].isin(tst_cids)]

    # AGE:
    print(trn.groupby('child_key').first()['age_years'].mean(), trn.groupby('child_key').first()['age_years'].std(),
          tst.groupby('child_key').first()['age_years'].mean(), tst.groupby('child_key').first()['age_years'].std())
    # ADOS:
    ados_trn = trn[trn['repeat_instrument'].apply(lambda i: 'ADOS' in i)].groupby('child_key').first()
    ados_tst = tst[tst['repeat_instrument'].apply(lambda i: 'ADOS' in i)].groupby('child_key').first()
    print('ADOS Total', ados_trn['x2'].astype(float).mean(), ados_trn['x2'].astype(float).std(), ados_tst['x2'].astype(float).mean(), ados_tst['x2'].astype(float).std())
    print('ADOS SA', ados_trn['x0'].astype(float).mean(), ados_trn['x0'].astype(float).std(), ados_tst['x0'].astype(float).mean(), ados_tst['x0'].astype(float).std())
    print('ADOS RRB', ados_trn['x1'].astype(float).mean(), ados_trn['x1'].astype(float).std(), ados_tst['x1'].astype(float).mean(), ados_tst['x1'].astype(float).std())
    print('ADOS D2', ados_trn['x3'].astype(float).mean(), ados_trn['x3'].astype(float).std(), ados_tst['x3'].astype(float).mean(), ados_tst['x3'].astype(float).std())
    d4_trn = ados_trn.apply(lambda r: r['x4'] if 'Toddlers' not in r['repeat_instrument'] else r['x5'], axis=1)
    d4_tst = ados_tst.apply(lambda r: r['x4'] if 'Toddlers' not in r['repeat_instrument'] else r['x5'], axis=1)
    print('ADOS D4', d4_trn.astype(float).mean(), d4_trn.astype(float).std(), d4_tst.astype(float).mean(), d4_tst.astype(float).std())
    # COGNITIVE
    cog_trn = trn[trn['repeat_instrument'].apply(lambda i: 'Cognitive' in i)].groupby('child_key').first()
    cog_tst = tst[tst['repeat_instrument'].apply(lambda i: 'Cognitive' in i)].groupby('child_key').first()
    print('Cognitive', cog_trn['x1'].astype(float).mean(), cog_trn['x1'].astype(float).std(), cog_tst['x1'].astype(float).mean(), cog_tst['x1'].astype(float).std())
    # PLS
    pls_trn = trn[trn['repeat_instrument'].apply(lambda i: 'PLS' in i)].groupby('child_key').first()
    pls_tst = tst[tst['repeat_instrument'].apply(lambda i: 'PLS' in i)].groupby('child_key').first()
    print('PLS', pls_trn['x0'].astype(float).mean(), pls_trn['x0'].astype(float).std(), pls_tst['x0'].astype(float).mean(), pls_tst['x0'].astype(float).std())

    # human = df[df['annotator'] != NET_NAME].copy()
    model_statistics([df], ['Human'])

    print()
