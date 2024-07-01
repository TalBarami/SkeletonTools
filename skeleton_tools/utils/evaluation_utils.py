import os
from os import path as osp

import numpy as np
import pandas as pd
import seaborn as sns
from skeleton_tools.utils.constants import NET_NAME, DB_PATH, read_db
from skeleton_tools.utils.tools import read_pkl, get_video_properties, init_directories, read_json, write_json

pd.set_option('display.expand_frame_repr', False)
sns.set_theme()


def unify(df):
    _df = pd.DataFrame(columns=df.columns)

    def calc_weighted_mean_score(rows, col):
        weights = [(row['end_frame'] - row['start_frame']) / (rows[-1]['end_frame'] - rows[0]['start_frame']) for row in rows]
        scores = [row[col] for row in rows]
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
                                     merge[0]['movement'], merge[0]['calc_date'], merge[0]['annotator'], calc_weighted_mean_score(merge, 'stereotypical_score'), calc_weighted_mean_score(merge, 'no_action_score')]
        else:
            start_time = _df.iloc[_df.shape[0] - 1]['end_time'] if i > 0 else curr['start_time']
            end_time = df.iloc[i + 1]['start_time'] if i < n - 1 else curr['end_time']
            start_frame = _df.iloc[_df.shape[0] - 1]['end_frame'] if i > 0 else curr['start_frame']
            end_frame = df.iloc[i + 1]['start_frame'] if i < n - 1 else curr['end_frame']
            _df.loc[_df.shape[0]] = [curr['video'], curr['video_full_name'], curr['video_path'], start_time, end_time, start_frame, end_frame,
                                     curr['movement'], curr['calc_date'], curr['annotator'], curr['stereotypical_score'], curr['no_action_score']]
        i += 1
    return _df


def aggregate(df, threshold=None):
    _df = pd.DataFrame(columns=df.columns)
    if threshold:
        df['prediction'] = np.where(df['stereotypical_score'] > threshold, 'Stereotypical', 'NoAction')
    else:
        df['prediction'] = np.where(df['stereotypical_score'] > df['no_action_score'], 'Stereotypical', 'NoAction')
    i = 0
    while i < df.shape[0]:
        r = df.iloc[i]
        s, t, c, n, p = r['start_frame'], r['end_frame'], r['stereotypical_score'], r['no_action_score'], r['prediction']
        _s, _t = r['start_time'], r['end_time']
        score = [c]
        neg_score = [n]
        j = i + 1
        while j < df.shape[0]:
            rr = df.iloc[j]
            ss, tt, cc, nn, pp = rr['start_frame'], rr['end_frame'], rr['stereotypical_score'], rr['no_action_score'], rr['prediction']
            if p == pp:
                t = tt
                _t = rr['end_time']
                score.append(cc)
                neg_score.append(nn)
            else:
                break
            j += 1
        _df.loc[_df.shape[0]] = [r['video'], r['video_full_name'], r['video_path'], _s, _t, s, t, p, pd.Timestamp.now(), r['annotator'], np.mean(score), np.mean(neg_score)]
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


def evaluate(_df, ground_truth, key='frame'):
    if _df.empty:
        return 0, 0, 0, 0
    df = _df.copy()
    df['movement'] = df['movement'].apply(lambda s: 0 if 'NoAction' in s else 1)
    df['segment_length'] = df[f'end_{key}'] - df[f'start_{key}']
    df['hit_score'] = 0
    df['miss_score'] = 0
    human_intervals = ground_truth[ground_truth['video'] == df['video'].values[0]][[f'start_{key}', f'end_{key}']].values.tolist()

    for i, row in df.iterrows():
        model_interval = row[[f'start_{key}', f'end_{key}']].values.tolist()
        intersections = [(i, get_intersection(model_interval, human_interval), human_interval) for i, human_interval in enumerate(human_intervals)]
        intersections = [(i, m, h) for i, m, h in intersections if m is not None]
        slength = row['segment_length']
        if row['movement'] == 1:
            hit = np.sum([t-s for _, _, (s, t) in intersections])
            miss = slength - hit
            # remove from human intervals:
            human_intervals = [h for i, h in enumerate(human_intervals) if i not in [j for j, _, _ in intersections]]
            # hit, miss = (slength, 0) if len(intersections) > 0 else (0, slength)
        else:
            punish = [intersect for _, intersect, interval in intersections if intersect == interval]
            miss = np.sum([min(high, model_interval[1]) - max(low, model_interval[0]) for (low, high) in punish])
            hit = slength - miss
        df.loc[i, ['hit_score', 'miss_score']] = hit, miss

    tp = df[df['movement'] == 1]['hit_score'].sum()
    fp = df[df['movement'] == 1]['miss_score'].sum()
    fn = df[df['movement'] == 0]['miss_score'].sum()
    tn = df[df['movement'] == 0]['hit_score'].sum()

    # print(f'{df.iloc[0]["video"]}: {tp} {fp} {tn} {fn}')
    return tp, fp, fn, tn


def evaluate_threshold(score_files, human_labels, thresholds, per_assessment=False):
    if per_assessment:
        human_labels = pd.concat([aggregate_cameras(g, fillna=False) for _, g in human_labels.groupby('assessment')])
    dfs = pd.concat([pd.read_csv(fp) for f, fp in score_files.values])
    df = pd.DataFrame(columns=['video', 'threshold', 'tp', 'fp', 'fn', 'tn', 'frame_count'])
    for t in thresholds:
        # print(f'Threshold: {t}')
        # out_file = osp.join(out_path, f'predictions_{t}.csv')
        # if osp.exists(out_file):
        #     print(f'Loading predictions from {out_file}...')
        #     agg = pd.read_csv(out_file)
        # else:
        # print(f'Preparing dataframes...')
        agg = pd.concat([aggregate(df, t) for _, df in dfs.groupby('video')])
        agg['assessment'] = agg['video'].apply(lambda s: '_'.join(s.split('_')[:-2]))
        agg['child_key'] = agg['assessment'].apply(lambda s: s.split('_')[0])
        frames_counts = agg.groupby('video')['end_frame'].max().reset_index()
        frames_counts.columns = ['video', 'frame_count']
        agg = pd.merge(agg, frames_counts, on='video', how='left')
            # agg.to_csv(out_file, index=False)
        n, m = agg['video'].nunique(), agg['assessment'].nunique()
        if per_assessment:
            agg = [aggregate_cameras(g, fillna=True) for _, g in agg.groupby('assessment')]
        else:
            agg = [df.reset_index() for _, df in agg.groupby('video')]
        for _df in agg:
            tp, fp, fn, tn = evaluate(_df, human_labels, key='frame')
            df.loc[df.shape[0]] = [_df['video'].iloc[0], t, tp, fp, fn, tn, _df['frame_count'].iloc[0]]
        _df = df[df['threshold'] == t]
        tp, fp, fn, tn = np.sum(_df[['tp', 'fp', 'fn', 'tn']].values, axis=0)
        accuracy, precision, recall = (tp + tn) / (tp + tn + fp + fn), tp / (tp + fp), tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        print(f'\t{t}: F1={f1}, Accuracy={accuracy}, Precision={precision}, Recall={recall} (Total {m} assessments over {n} videos, per_assessment={per_assessment})')
    return df

def evaluate_model(score_files, human_labels, per_assessment=False):
    if per_assessment:
        human_labels = pd.concat([aggregate_cameras(g, fillna=False) for _, g in human_labels.groupby('assessment')])
    agg = pd.concat([pd.read_csv(fp) for f, fp in score_files.values])
    df = pd.DataFrame(columns=['video', 'tp', 'fp', 'fn', 'tn', 'frame_count'])
    agg['assessment'] = agg['video'].apply(lambda s: '_'.join(s.split('_')[:-2]))
    agg['child_key'] = agg['assessment'].apply(lambda s: s.split('_')[0])
    frames_counts = agg.groupby('video')['end_frame'].max().reset_index()
    frames_counts.columns = ['video', 'frame_count']
    agg = pd.merge(agg, frames_counts, on='video', how='left')
    n, m, c = agg['video'].nunique(), agg['assessment'].nunique(), agg['child_key'].nunique()
    if per_assessment:
        agg = [aggregate_cameras(g, fillna=True) for _, g in agg.groupby('assessment')]
    else:
        agg = [df.reset_index() for _, df in agg.groupby('video')]
    for _df in agg:
        tp, fp, fn, tn = evaluate(_df, human_labels, key='frame')
        df.loc[df.shape[0]] = [_df['video'].iloc[0], tp, fp, fn, tn, _df['frame_count'].iloc[0]]
    tp, fp, fn, tn = np.sum(df[['tp', 'fp', 'fn', 'tn']].values, axis=0)
    accuracy, precision, recall = (tp + tn) / (tp + tn + fp + fn), tp / (tp + fp), tp / (tp + fn)
    print(f'\tAccuracy={accuracy}, Precision={precision}, Recall={recall} (Total {m} assessments over {n} videos of {c} children. per_assessment={per_assessment}).')
    return df

def aggregate_table(df):
    df = df[df['movement'] != 'NoAction']
    df['stereotypical_length'] = (df['end_time'] - df['start_time'])
    df['stereotypical_relative_length'] = df['stereotypical_length'] / df['length_seconds']
    table = df.groupby(['video', 'annotator']).agg({'length_seconds': 'mean', 'stereotypical_length': ['sum', 'mean', 'count'], 'stereotypical_relative_length': ['sum', 'mean']})
    table.columns = ['_'.join(col).strip() for col in table.columns.values]
    table = table.reset_index()
    # out = pd.DataFrame(columns=['video', 'length_seconds', 'count_stereotypical', 'sum_stereotypical_length', 'mean_stereotypical_length', 'sum_stereotypical_relative_length', 'mean_stereotypical_relative_length'])
    # out['video'] = table.index
    # out['length_seconds'] = table['length_seconds']['mean'].values
    # out[['sum_stereotypical_length', 'mean_stereotypical_length', 'count_stereotypical']] = table['stereotypical_length'].values
    # out[['sum_stereotypical_relative_length', 'mean_stereotypical_relative_length']] = table['stereotypical_relative_length'].values
    return table


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


def aggregate_cameras(agg_df, fillna=False):
    frame_count = agg_df['frame_count'].max()
    # annotator = agg_df['annotator'].iloc[0]
    dfs = [g for v, g in agg_df.groupby('video')]
    out = pd.DataFrame(columns=['assessment', 'start_frame', 'end_frame', 'movement'])
    for df in dfs:
        df = df[df['movement'] != 'NoAction']
        for i, row in df.iterrows():
            intersection = [j for j, r in out.iterrows() if get_intersection((row['start_frame'], row['end_frame']), (r['start_frame'], r['end_frame'])) is not None]
            if len(intersection) > 0:
                _df = pd.DataFrame([row] + [out.loc[j] for j in intersection])
                row[['start_frame', 'end_frame']] = (_df['start_frame'].min(), _df['end_frame'].max())
                out = out.drop(intersection).reset_index(drop=True)
            out.loc[out.shape[0]] = row[out.columns]
    out = out.sort_values(by=['assessment', 'start_frame']).reset_index(drop=True)
    if fillna:
        assessment = agg_df.iloc[0]['assessment']
        out_na = pd.DataFrame(columns=out.columns)
        if out.empty:
            out_na.loc[out_na.shape[0]] = (assessment, 0, frame_count, 'NoAction')
        else:
            n = out.shape[0]
            for i in range(n-1):
                curr_row = out.loc[i]
                next_row = out.loc[i+1]
                s, t = curr_row['end_frame'], next_row['start_frame']
                out_na.loc[out_na.shape[0]] = (assessment, s, t, 'NoAction')
            if out.loc[0]['start_frame'] > 0:
                out_na.loc[out_na.shape[0]] = (assessment, 0, out.loc[0]['start_frame'], 'NoAction')
            if out.loc[n-1]['end_frame'] < frame_count:
                out_na.loc[out_na.shape[0]] = (assessment, out.loc[n-1]['end_frame'], frame_count, 'NoAction')
        out = pd.concat([out, out_na])
    out['frame_count'] = frame_count
    out['video'] = out['assessment']
    # out['annotator'] = annotator
    out = out.sort_values(by=['assessment', 'start_frame']).reset_index(drop=True)
    return out


def aggregate_cameras_for_annotations(annotations, fillna=False):
    groups = list(annotations.groupby('assessment'))
    result = []
    for assessment, group in groups:
        result.append([aggregate_cameras(g, fillna=fillna) for _, g in group.groupby('annotator')])
    result = [d for dfs in result for d in dfs]
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


def collect_labels(root, model_name, file_extension='annotations', out=None, subset=None):
    names = [f for f in os.listdir(root)]
    if subset is not None:
        names = [f for f in names if f in subset]
    files = [osp.join(root, f, model_name, f'{f}_{file_extension}.csv') for f in names]
    dfs = [pd.read_csv(f) for f in files if osp.exists(f)]
    df = pd.concat(dfs)
    df['assessment'] = df['video'].apply(lambda s: '_'.join(s.split('_')[:-2]))
    df['child_id'] = df['assessment'].apply(lambda s: s.split('_')[0])
    df = df.sort_values(by=['assessment', 'start_time'] if 'start_time' in df.columns else ['assessment'])
    if out:
        df.to_csv(out, index=False)
    return df

def collect_predictions(predictions_dir, experiment_name=None, out_dir=None, model_name=None, subset=None):
    if model_name is None:
        name = experiment_name
        df = pd.read_csv(osp.join(predictions_dir, f'{experiment_name}.csv'))
    else:
        name = model_name if experiment_name is None else f'{experiment_name}_{model_name}'
        df = collect_labels(predictions_dir, osp.join('jordi', model_name))
    if subset is not None:
        df = df[df['video'].isin(subset)]
    # manual_fix = {
    #     '671336821_ADOS_Clinical_220118': 2268
    # }
    df = prepare(df.reset_index())
    # df['length_seconds'] = df.apply(lambda row: manual_fix[row['assessment']] if row['assessment'] in manual_fix.keys() else row['length_seconds'], axis=1)
    summary_df, assessment_df, summary_assessment_df = generate_aggregations(df)

    if out_dir is not None:
        df.to_csv(osp.join(out_dir, f'base_{name}.csv'), index=False)
        summary_df.to_csv(osp.join(out_dir, f'summary_{name}.csv'), index=False)
        assessment_df.to_csv(osp.join(out_dir, f'assessment_{name}.csv'), index=False)
        summary_assessment_df.to_csv(osp.join(out_dir, f'summary_assessment_{name}.csv'), index=False)
    return df, summary_df, assessment_df, summary_assessment_df

def prepare(df, remove_noact=False):
    if remove_noact:
        df = df[df['movement'] != 'NoAction']
    db = read_db()
    drop_cols = ['width', 'height', 'fps', 'frame_count', 'length_seconds', 'length', 'total_frames', 'assessment', 'child_id', 'video_path']
    df['annotator'] = df['annotator'].apply(lambda a: NET_NAME if a == NET_NAME else 'Human')
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    missing = df[~df['video'].isin(db['basename'])]
    if not missing.empty:
        for k in [c for c in ['video', 'video_full_name', 'segment_name'] if c in missing.columns]:
            df.loc[missing.index, k] = missing['video'].apply(lambda v: v.replace('Clinical', 'Control') if 'Clinical' in v else v.replace('Control', 'Clinical'))

    df = pd.merge(df, db, left_on='video', right_on='basename', how='left')
    # df[['width', 'height', 'fps', 'frame_count', 'length_seconds']] = df.apply(lambda row: db[db['video'] == row['video_full_name']].iloc[0][['width', 'height', 'fps', 'frame_count', 'length_seconds']], axis=1)
    # df[['resolution', 'fps', 'total_frames', 'length_seconds']] = df.apply(lambda row: db[db['final_name'] == row['video_full_name']].iloc[0][['fixed_resolution', 'fixed_fps', 'fixed_total_frames', 'fixed_length']], axis=1)
    # df['assessment                        '] = df['video'].apply(lambda v: '_'.join(v.split('_')[:-2]))
    return df

def generate_aggregations(df):
    summary_df = aggregate_table(df)
    assessment_df = aggregate_cameras_for_annotations(df)
    summary_assessment_df = aggregate_table(assessment_df)

    return summary_df, assessment_df, summary_assessment_df