import os
from os import path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skeleton_tools.utils.constants import NET_NAME, REMOTE_STORAGE, DB_PATH

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
            _df.loc[_df.shape[0]] = [merge[0]['video'], merge[0]['start_time'], merge[-1]['end_time'], merge[0]['start_frame'], merge[-1]['end_frame'],
                                     merge[0]['movement'], merge[0]['calc_date'], merge[0]['annotator'], calc_weighted_mean_score(merge)]
        else:
            start_time = _df.iloc[_df.shape[0] - 1]['end_time'] if i > 0 else curr['start_time']
            end_time = df.iloc[i + 1]['start_time'] if i < n - 1 else curr['end_time']
            start_frame = _df.iloc[_df.shape[0] - 1]['end_frame'] if i > 0 else curr['start_frame']
            end_frame = df.iloc[i + 1]['start_frame'] if i < n - 1 else curr['end_frame']
            _df.loc[_df.shape[0]] = [curr['video'], start_time, end_time, start_frame, end_frame,
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
        _df.loc[_df.shape[0]] = [r['video'], _s, _t, s, t, p, pd.Timestamp.now(), NET_NAME, np.mean(score)]
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

def evaluate(df, ground_truth, min_iou=0.1, key='frame'):
    df = df.copy()
    df['movement'] = df['movement'].apply(lambda s: 0 if 'NoAction' in s else 1)
    df['segment_length'] = df[f'end_{key}'] - df[f'start_{key}']
    # model = df[df['annotator'] == NET_NAME]
    model = df
    model['hit_score'] = 0
    model['miss_score'] = 0
    human_intervals = ground_truth[ground_truth['video'] == df['video'].values[0]][[f'start_{key}', f'end_{key}']].values.tolist()

    for i, row in model.iterrows():
        model_interval = row[[f'start_{key}', f'end_{key}']].values.tolist()
        movement = row['movement']
        ious = [x for x in (iou_1d(model_interval, human_interval) for human_interval in human_intervals) if x is not None]
        if movement == 1:
            miss, hit = row['segment_length'], 0
            if len(ious) > 0 and any(iou > min_iou for iou in ious):
                hit, miss = miss, hit
        else:
            intersections = [x for x in ((get_intersection(model_interval, human_interval), human_interval) for human_interval in human_intervals) if x[0] is not None]
            intersections = [intersect for intersect, interval in intersections if intersect == interval]
            miss = np.sum([high - low for (low, high) in intersections])
            hit = row['segment_length'] - miss
        model.loc[i, ['hit_score', 'miss_score']] = hit, miss

    total_length = model[f'end_{key}'].max()
    tp = model[model['movement'] == 1]['hit_score'].sum()
    fp = model[model['movement'] == 1]['miss_score'].sum()
    fn = model[model['movement'] == 0]['miss_score'].sum()
    tn = model[model['movement'] == 0]['hit_score'].sum()

    # conf_mat = np.array([[tp, fp], [fn, tn]]) / total_length
    # precision, recall = tp / (tp + fp), tp / (tp + fn)

    return tp, fp, fn, tn

def evaluate_threshold(score_files, human_labels, per_assessment=False):
    dfs = [pd.read_csv(p) for p in score_files]
    thresholds = np.round(np.arange(0.05, 1, 0.05), 3)
    c, p, r = [], [], []
    for t in thresholds:
        print(f'Threshold: {t}')
        agg = [aggregate(df.copy(), t) for df in dfs]
        if per_assessment:
            agg = pd.concat(agg)
            agg = prepare(agg)
            agg = [aggregate_cameras(g, fillna=True) for _, g in agg.groupby('assessment')]
        tp, fp, fn, tn = np.sum(list(zip(*[evaluate(df, human_labels, key='time') for df in agg])), axis=1)
        precision, recall = tp / (tp + fp), tp / (tp + fn)
        # conf_mat, precision, recall = list(zip(*[evaluate(df, human_labels) for df in agg]))
        # c.append(np.nanmean(conf_mat, axis=0))
        p.append(np.nanmean(precision))
        r.append(np.nanmean(recall))
    return thresholds, p, r

def aggregate_table(df):
    df['stereotypical_length'] = (df['end_time'] - df['start_time'])
    df['stereotypical_relative_length'] = df['stereotypical_length'] / df['length_seconds']
    table = df.groupby('video').agg({'length_seconds': 'mean', 'stereotypical_length': ['sum', 'mean', 'count'], 'stereotypical_relative_length': ['sum', 'mean']})
    out = pd.DataFrame(columns=['video', 'length_seconds', 'count_stereotypical', 'sum_stereotypical_length', 'mean_stereotypical_length', 'sum_stereotypical_relative_length', 'mean_stereotypical_relative_length'])
    out['video'] = table.index
    out['length_seconds'] = table['length_seconds']['mean'].values
    out[['sum_stereotypical_length', 'mean_stereotypical_length', 'count_stereotypical']] = table['stereotypical_length'].values
    out[['sum_stereotypical_relative_length', 'mean_stereotypical_relative_length']] = table['stereotypical_relative_length'].values
    return out

def aggregate_cameras(df, fillna=False):
    length = df['end_time'].max()
    dfs = [g for _, g in df.groupby('video')]
    out = pd.DataFrame(columns = ['assessment', 'start_time', 'end_time', 'movement'])
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

def intersect(*lst):
    names = [set(df['video'].unique()) for df in lst]
    names = set.intersection(*names)
    out = [df[df['video'].isin(names)] for df in lst]
    return out

def prepare(df, remove_noact=False):
    if remove_noact:
        df = df[df['movement'] != 'NoAction']
    db = pd.read_csv(DB_PATH)
    df[['resolution', 'fps', 'total_frames', 'length_seconds']] = df.apply(lambda row: db[db['final_name'] == row['video']].iloc[0][['fixed_resolution', 'fixed_fps', 'fixed_total_frames', 'fixed_length']], axis=1)
    df['assessment'] = df['video'].apply(lambda v: '_'.join(v.split('_')[:-2]))
    return df

if __name__ == '__main__':
    root = osp.join(REMOTE_STORAGE, r'Users\TalBarami\JORDI_50_vids_benchmark\JORDIv3')
    human_labels1 = pd.read_csv(osp.join(REMOTE_STORAGE, r'Users\TalBarami\JORDI_50_vids_benchmark\annotations\human_labels.csv'))
    human_labels2 = pd.read_csv(osp.join(REMOTE_STORAGE, r'Users\TalBarami\JORDI_50_vids_benchmark\annotations\labels_post_qa.csv'))
    names = human_labels2['video'].apply(lambda v: osp.splitext(v)[0]).unique()
    human_labels1 = human_labels1[human_labels1['video'].apply(lambda v: osp.splitext(v)[0]).isin(names)]
    files = [x for x in (osp.join(root, f, 'binary_weighted_extra_noact_epoch_18.pth', f'{f}_scores.csv') for f in os.listdir(root) if f in names) if osp.exists(x)]
    t1, p1, r1 = evaluate_threshold(files, human_labels1)
    t2, p2, r2 = evaluate_threshold(files, human_labels2)
    df1 = pd.DataFrame(columns=['threshold', 'precision', 'recall'], data=np.array([t1, p1, r1]).T)
    df1['annotations'] = 'Old'
    df2 = pd.DataFrame(columns=['threshold', 'precision', 'recall'], data=np.array([t2, p2, r2]).T)
    df2['annotations'] = 'New'
    df = pd.concat((df1, df2)).reset_index(drop=True)
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x='recall', y='precision', marker='o', hue='annotations')
    for _, row in df.iterrows():
        ax.text(x=row['recall'] + 0.005, y=row['precision'] + 0.005, s=row['threshold'],
                bbox=dict(facecolor='lightblue', alpha=0.5))
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    plt.show()

