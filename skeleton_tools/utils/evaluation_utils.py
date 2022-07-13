import os
from os import path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skeleton_tools.utils.constants import NET_NAME
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
            while j < df.shape[0] - 2:
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

def evaluate(df, ground_truth, min_iou=0.1):
    df = df.copy()
    df['movement'] = df['movement'].apply(lambda s: 0 if 'NoAction' in s else 1)
    df['segment_length'] = df['end_frame'] - df['start_frame']
    model = df[df['annotator'] == NET_NAME]
    model['hit_score'] = 0
    model['miss_score'] = 0
    human_intervals = ground_truth[ground_truth['video'] == df['video'].values[0]][['start_frame', 'end_frame']].values.tolist()

    for i, row in model.iterrows():
        try:
            model_interval = row[['start_frame', 'end_frame']].values.tolist()
            movement = row['movement']
            ious = [x for x in (iou_1d(model_interval, human_interval) for human_interval in human_intervals) if x is not None]
            if movement == 1:
                miss, hit = row['segment_length'], 0
                if len(ious) > 0 and any(iou > min_iou for iou in ious):
                    hit, miss = miss, hit
                # if len(ious) > 1:
                #         print(f"WTF!!! {row['video']}")
            else:
                intersections = [x for x in ((get_intersection(model_interval, human_interval), human_interval) for human_interval in human_intervals) if x[0] is not None]
                intersections = [intersect for intersect, interval in intersections if intersect == interval]
                miss = np.sum([high - low for (low, high) in intersections])
                hit = row['segment_length'] - miss
            model.loc[i, ['hit_score', 'miss_score']] = hit, miss
        except Exception as e:
            print(1)

    total_length = model['end_frame'].max()
    tp = model[model['movement'] == 1]['hit_score'].sum()
    fp = model[model['movement'] == 1]['miss_score'].sum()
    fn = model[model['movement'] == 0]['miss_score'].sum()
    tn = model[model['movement'] == 0]['hit_score'].sum()

    # conf_mat = np.array([[tp, fp], [fn, tn]]) / total_length
    # precision, recall = tp / (tp + fp), tp / (tp + fn)

    return tp, fp, fn, tn

def evaluate_threshold(scores, human_labels):
    dfs = [pd.read_csv(p) for p in scores]
    thresholds = np.round(np.arange(0.4, 1, 0.05), 3)
    c, p, r = [], [], []
    for t in thresholds:
        print(f'Threshold: {t}')
        agg = [aggregate(df.copy(), t) for df in dfs]
        tp, fp, fn, tn = np.sum(list(zip(*[evaluate(df, human_labels) for df in agg])), axis=1)
        precision, recall = tp / (tp + fp), tp / (tp + fn)
        # conf_mat, precision, recall = list(zip(*[evaluate(df, human_labels) for df in agg]))
        # c.append(np.nanmean(conf_mat, axis=0))
        p.append(np.nanmean(precision))
        r.append(np.nanmean(recall))

    df = pd.DataFrame(columns=['threshold', 'precision', 'recall'], data=np.array([thresholds, p, r]).T)
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x='recall', y='precision', marker='o')
    for _, row in df.iterrows():
        ax.text(x=row['recall'] + 0.005, y=row['precision'] + 0.005, s=row['threshold'], bbox=dict(facecolor='lightblue', alpha=0.5))
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    plt.show()

if __name__ == '__main__':
    root = r'Z:\Users\TalBarami\JORDI_50_vids_benchmark\JORDIv3'
    human_labels = pd.read_csv(r'Z:\Users\TalBarami\JORDI_50_vids_benchmark\human_labels.csv')
    names = human_labels['video'].apply(lambda v: osp.splitext(v)[0]).unique()
    files = [x for x in (osp.join(root, f, 'binary_weighted_extra_noact_epoch_18.pth', f'{f}_scores.csv') for f in os.listdir(root) if f in names) if osp.exists(x)]
    evaluate_threshold(files, human_labels)