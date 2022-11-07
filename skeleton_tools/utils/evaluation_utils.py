import os
from os import path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skeleton_tools.utils.constants import NET_NAME, REMOTE_STORAGE, DB_PATH
from skeleton_tools.utils.tools import read_pkl, get_video_properties, init_directories, read_json, write_json

pd.set_option('display.expand_frame_repr', False)
sns.set_theme()


def unify(df):
    _df = pd.DataFrame(columns=df.columns)

    def calc_weighted_mean_score(rows):
        weights = [(row['end_frame'] - row['start_frame']) / (rows[-1]['end_frame'] - rows[0]['start_frame']) for row in rows]
        scores = [row['stereotypical_score'] for row in rows]
        return np.average(scores, weights=weights)

    n = df.shape[0]
    i = 0
    while i < n:
        curr = df.iloc[i]
        if curr['movement'] == 'Stereotypical':
            merge = [curr]
            j = i + 2
            while j < df.shape[0] - 1:
                next_row = df.iloc[j]
                if next_row['start_frame'] < curr['end_frame']:
                    merge.append(next_row)
                    curr = next_row
                    j += 2
                    i += 2
                else:
                    break
            _df.loc[_df.shape[0]] = [merge[0]['video'], merge[0]['video_full_name'], merge[0]['video_path'], merge[0]['start_time'], merge[-1]['end_time'], merge[0]['start_frame'], merge[-1]['end_frame'],
                                     merge[0]['movement'], merge[0]['calc_date'], merge[0]['annotator'], calc_weighted_mean_score(merge)]
        else:
            start_time = _df.iloc[_df.shape[0] - 1]['end_time'] if i > 0 else curr['start_time']
            end_time = df.iloc[i + 1]['start_time'] if i < n - 1 else curr['end_time']
            start_frame = _df.iloc[_df.shape[0] - 1]['end_frame'] if i > 0 else curr['start_frame']
            end_frame = df.iloc[i + 1]['start_frame'] if i < n - 1 else curr['end_frame']
            _df.loc[_df.shape[0]] = [curr['video'], curr['video_full_name'], curr['video_path'], start_time, end_time, start_frame, end_frame,
                                     curr['movement'], curr['calc_date'], curr['annotator'], curr['stereotypical_score']]
        i += 1
    return _df


def aggregate(df, threshold):
    _df = pd.DataFrame(columns=df.columns)
    df['prediction'] = np.where(df['stereotypical_score'] > threshold, 'Stereotypical', 'NoAction')
    i = 0
    while i < df.shape[0]:
        r = df.iloc[i]
        s, t, c, p = r['start_frame'], r['end_frame'], r['stereotypical_score'], r['prediction']
        _s, _t = r['start_time'], r['end_time']
        score = [c]
        j = i + 1
        while j < df.shape[0]:
            rr = df.iloc[j]
            ss, tt, cc, pp = rr['start_frame'], rr['end_frame'], rr['stereotypical_score'], rr['prediction']
            if p == pp:
                t = tt
                _t = rr['end_time']
                score.append(cc)
            else:
                break
            j += 1
        _df.loc[_df.shape[0]] = [r['video'], r['video_full_name'], r['video_path'], _s, _t, s, t, p, pd.Timestamp.now(), NET_NAME, np.mean(score)]
        i = j
    return unify(_df)


# def aggregate_b(self, df):  # TODO: Decide if using
#     _df = pd.DataFrame(columns=df.columns)
#     for i in range(0, df['end_frame'].max(), 30):
#         sdf = df[(df['start_frame'] <= i) & (i < df['end_frame'])]
#         _df.loc[_df.shape[0]] = [df['video'].loc[0], -1, -1, i, i + 30, -1, pd.Timestamp.now(), 'JORDI', sdf['stereotypical_score'].mean()]
#     return _df

def get_intersection(interval1, interval2):
    new_min = max(interval1[0], interval2[0])
    new_max = min(interval1[1], interval2[1])
    return [new_min, new_max] if new_min <= new_max else None


def get_union(interval1, interval2):
    new_min = min(interval1[0], interval2[0])
    new_max = max(interval1[1], interval2[1])
    return [new_min, new_max] if new_min <= new_max else None


def iou_1d(interval1, interval2):
    intersection = get_intersection(interval1, interval2)
    if intersection is None:
        return None
    union = get_union(interval1, interval2)
    iou = (intersection[1] - intersection[0]) / (union[1] - union[0])
    return iou


def evaluate(df, ground_truth, key='frame'):
    if df.empty:
        return 0, 0, 0, 0
    df = df.copy()
    df['movement'] = df['movement'].apply(lambda s: 0 if 'NoAction' in s else 1)
    df['segment_length'] = df[f'end_{key}'] - df[f'start_{key}']
    df['hit_score'] = 0
    df['miss_score'] = 0
    human_intervals = ground_truth[ground_truth['video'] == df['video'].values[0]][[f'start_{key}', f'end_{key}']].values.tolist()

    for i, row in df.iterrows():
        model_interval = row[[f'start_{key}', f'end_{key}']].values.tolist()
        intersections = [x for x in ((get_intersection(model_interval, human_interval), human_interval) for human_interval in human_intervals) if x[0] is not None]
        slength = row['segment_length']
        if row['movement'] == 1:
            hit, miss = (slength, 0) if len(intersections) > 0 else (0, slength)
        else:
            intersections = [intersect for intersect, interval in intersections if intersect == interval]
            miss = np.sum([high - low for (low, high) in intersections])
            hit = slength - miss
        df.loc[i, ['hit_score', 'miss_score']] = hit, miss

    tp = df[df['movement'] == 1]['hit_score'].sum()
    fp = df[df['movement'] == 1]['miss_score'].sum()
    fn = df[df['movement'] == 0]['miss_score'].sum()
    tn = df[df['movement'] == 0]['hit_score'].sum()

    # print(f'{df.iloc[0]["video"]}: {tp} {fp} {tn} {fn}')
    return tp, fp, fn, tn


def evaluate_threshold(score_files, human_labels, out_path, per_assessment=False):
    init_directories(out_path)
    dfs = pd.concat([pd.read_csv(p) for p in score_files])
    thresholds = np.round(np.arange(0.35, 1, 0.05), 3)
    a, p, r = [], [], []
    for t in thresholds:
        print(f'Threshold: {t}')
        out_file = osp.join(out_path, f'predictions_{t}.csv')
        if osp.exists(out_file):
            agg = pd.read_csv(out_file)
        else:
            print(f'Preparing dataframes...')
            agg = pd.concat([prepare(aggregate(df, t)) for _, df in dfs.groupby('video')])
            agg.to_csv(out_file, index=False)
        n, m = agg['video'].nunique(), agg['assessment'].nunique()
        if per_assessment:
            agg = [aggregate_cameras(g, fillna=True) for _, g in agg.groupby('assessment')]
        else:
            agg = [df for _, df in agg.groupby('video')]
        tp, fp, fn, tn = np.sum(list(zip(*[evaluate(df, human_labels, key='time') for df in agg])), axis=1)
        accuracy, precision, recall = (tp + tn) / (tp + tn + fp + fn), tp / (tp + fp), tp / (tp + fn)
        print(f'\tAccuracy={accuracy}, Precision={precision}, Recall={recall} (Total {m} assessments over {n} videos, per_assessment={per_assessment})')
        a.append(accuracy)
        p.append(precision)
        r.append(recall)
    return thresholds, a, p, r


def aggregate_table(df):
    df = df[df['movement'] != 'NoAction']
    df['stereotypical_length'] = (df['end_time'] - df['start_time'])
    df['stereotypical_relative_length'] = df['stereotypical_length'] / df['length_seconds']
    table = df.groupby('video').agg({'length_seconds': 'mean', 'stereotypical_length': ['sum', 'mean', 'count'], 'stereotypical_relative_length': ['sum', 'mean']})
    out = pd.DataFrame(columns=['video', 'length_seconds', 'count_stereotypical', 'sum_stereotypical_length', 'mean_stereotypical_length', 'sum_stereotypical_relative_length', 'mean_stereotypical_relative_length'])
    out['video'] = table.index
    out['length_seconds'] = table['length_seconds']['mean'].values
    out[['sum_stereotypical_length', 'mean_stereotypical_length', 'count_stereotypical']] = table['stereotypical_length'].values
    out[['sum_stereotypical_relative_length', 'mean_stereotypical_relative_length']] = table['stereotypical_relative_length'].values
    return out


adjustments = read_json(r'Z:\Users\TalBarami\JORDI_50_vids_benchmark\annotations\adjustments.json') if osp.exists(r'Z:\Users\TalBarami\JORDI_50_vids_benchmark\annotations\adjustments.json') else {}


def get_adjust(name):
    if name in adjustments.keys():
        return adjustments[name]
    else:
        basename = osp.splitext(name)[0]
        skel_path = osp.join(r'Z:\Users\TalBarami\JORDI_50_vids_benchmark\JORDIv3', basename, f'{basename}.pkl')
        if osp.exists(skel_path):
            T = read_pkl(skel_path)['keypoint'].shape[1]
            _, _, L, _ = get_video_properties(osp.join(r'Z:\Users\TalBarami\JORDI_50_vids_benchmark\videos', name.split('_')[0], name), method='cv2')
            adj = L - T
            adjustments[name] = adj
            write_json(adjustments, r'Z:\Users\TalBarami\JORDI_50_vids_benchmark\annotations\adjustments.json')
            return adj
        else:
            print(f'No skeleton for {name}')
            return None


def aggregate_cameras(df, fillna=False):
    length = df['length_seconds'].max()
    dfs = [g for v, g in df.groupby('video')]
    out = pd.DataFrame(columns=['assessment', 'start_time', 'end_time', 'movement'])
    for df in dfs:
        df = df[df['movement'] != 'NoAction']
        for i, row in df.iterrows():
            intersection = [j for j, r in out.iterrows() if get_intersection((row['start_time'], row['end_time']), (r['start_time'], r['end_time'])) is not None]
            if len(intersection) > 0:
                _df = pd.DataFrame([row] + [out.loc[j] for j in intersection])
                row[['start_time', 'end_time']] = (_df['start_time'].min(), _df['end_time'].max())
                out = out.drop(intersection).reset_index(drop=True)
            out.loc[out.shape[0]] = row[out.columns]
    out = out.sort_values(by=['assessment', 'start_time']).reset_index(drop=True)
    if fillna and not out.empty:
        assessment = out.loc[0]['assessment']
        n = out.shape[0]
        s = 0
        for i in range(n):
            curr_row = out.loc[i]
            out.loc[out.shape[0]] = (assessment, s, curr_row['start_time'], 'NoAction')
            s = curr_row['end_time']
        out.loc[out.shape[0]] = (assessment, s, length, 'NoAction')
    out['length_seconds'] = length
    out['video'] = out['assessment']
    out = out.sort_values(by=['assessment', 'start_time']).reset_index(drop=True)
    return out


def aggregate_cameras_for_annotations(annotations, fillna=False):
    groups = list(annotations.groupby('assessment'))
    result = []
    for assessment, group in groups:
        result.append(aggregate_cameras(group, fillna=fillna))
    result = pd.concat(result).sort_values(by=['assessment', 'start_time'])
    result['video'] = result['assessment']
    return result


def intersect(*lst, on='assessment', exclude=None):
    names = [set(df[on].unique()) for df in lst]
    names = set.intersection(*names)
    out = [df[df[on].isin(names)] for df in lst]
    if exclude is not None:
        out = [df[~df['assessment'].isin(exclude)] for df in out]
    return out


def prepare(df, remove_noact=False):
    if remove_noact:
        df = df[df['movement'] != 'NoAction']
    db = pd.read_csv(DB_PATH)
    df[['resolution', 'fps', 'total_frames', 'length_seconds']] = df.apply(lambda row: db[db['final_name'] == row['video']].iloc[0][['fixed_resolution', 'fixed_fps', 'fixed_total_frames', 'fixed_length']], axis=1)
    df['assessment'] = df['video'].apply(lambda v: '_'.join(v.split('_')[:-2]))
    return df
