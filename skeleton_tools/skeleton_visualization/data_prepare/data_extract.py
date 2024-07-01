from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from os import path as osp
import seaborn as sns

from skeleton_tools.openpose_layouts.body import COCO_LAYOUT
from skeleton_tools.openpose_layouts.face import PYFEAT_FACIAL
from skeleton_tools.skeleton_visualization.paint_components.frame_painters.base_painters import GlobalPainter
from skeleton_tools.skeleton_visualization.paint_components.frame_painters.local_painters import GraphPainter
from skeleton_tools.utils.skeleton_utils import bounding_box
from skeleton_tools.utils.tools import read_pkl, write_pkl


class VisualizerDataExtractor(ABC):
    def __init__(self, graph_layout, scale):
        self.graph_layout = graph_layout
        self.scale = scale
        self.epsilon = 1e-1
        self.cmap = sns.color_palette('bright', 25)

    @abstractmethod
    def _extract(self, cfg):
        pass

    def __call__(self, cfg):
        return self._extract(cfg)

    def _assign_labels(self, result, cids):
        M, T = result['landmarks'].shape[:2]
        result['label_text'] = np.array([['Child' if cids[t] == i else 'Adult' for t in range(T)] for i in range(M)])
        result['colors'] = np.array([[(255, 0, 0) if cids[t] == i else (0, 0, 255) for t in range(T)] for i in range(M)])
        if 'person_ids' in result.keys():
            pids = result['person_ids'].astype(int)
            result['person_colors'] = np.array([[self.cmap[pids[i, t]] for t in range(T)] for i in range(M)]) * 255
        # result['label_text'] = np.array([['Child' if i == 0 else 'Adult' for t in range(T)] for i in range(M)])
        # result['colors'] = np.array([[(255, 0, 0) if i == 0 else (0, 0, 255) for t in range(T)] for i in range(M)])


class MMPoseDataExtractor(VisualizerDataExtractor):
    def __init__(self, graph_layout=COCO_LAYOUT, scale=1):
        super().__init__(graph_layout, scale)

    def _gen_boxes(self, landmarks, landmarks_scores):
        M, T = landmarks.shape[:2]
        boxes = np.array([[bounding_box(landmarks[i, t], landmarks_scores[i, t]) for t in range(T)] for i in range(M)])
        return boxes

    def _extract(self, cfg):
        if cfg['paths']['processed'].endswith('.pkl'):
            cfg['skeleton_path'] = cfg['paths']['processed']
        else:
            cfg['skeleton_path'] = osp.join(cfg['paths']['processed'], cfg['name'], 'jordi', f'{cfg["name"]}.pkl')
        cfg['predictions_path'] = osp.join(cfg['paths']['processed'], cfg['name'], 'jordi', 'cv0.pth', f'{cfg["name"]}_scores.csv')
        skeleton_path, predictions_path = cfg['skeleton_path'], cfg['predictions_path']
        data = read_pkl(skeleton_path)
        landmarks, landmarks_scores, cids = data['keypoint'] * self.scale, data['keypoint_score'], data['child_ids'].astype(int)
        cids = self.fix_detections(landmarks, landmarks_scores, cids)
        pids = data['person_ids'] if 'person_ids' in data.keys() else (np.ones(landmarks.shape[:2], dtype=int) * -1)
        T = landmarks.shape[1]
        if osp.exists(predictions_path):
            scores = pd.read_csv(predictions_path)
            df = pd.read_csv(scores) if type(scores) == str else scores
            df['frame'] = (df['start_frame'] + df['end_frame']) // 2
            df.set_index('frame', inplace=True)
            _scores = df['stereotypical_score']
            scores = np.array([_scores.loc[i] if i in df.index else np.nan for i in range(T)])
            scores = np.expand_dims(pd.Series(scores).interpolate(method='polynomial', order=2, limit_direction='both').fillna(method='ffill').fillna(0).values, axis=1)
        else:
            scores = np.zeros((T, 1))
        result = {'landmarks': landmarks.astype(int), 'landmarks_scores': landmarks_scores,
                  'resolution': np.array(data['img_shape']).astype(int) * self.scale, 'fps': data['fps'], 'frame_count': data['total_frames'], 'duration_seconds': data['length_seconds'],
                  'video_path': data['video_path'], 'filename': data['frame_dir'],
                  'child_ids': cids, 'child_detection_scores': data['child_detected'], 'predictions': scores, 'person_ids': pids}


        face_joints = [k for k, v in self.graph_layout.joints().items() if any([s in v for s in self.graph_layout.face_joints()])]
        result['boxes'] = self._gen_boxes(landmarks, landmarks_scores).astype(int)
        result['face_boxes'] = self._gen_boxes(landmarks[:, :, face_joints], landmarks_scores[:, :, face_joints]).astype(int)
        self._assign_labels(result, cids)
        return result

    def fix_detections(self, landmarks, landmarks_scores, cids, k=15):
        new_cids = np.ones_like(cids) * -1
        new_cids[:k] = cids[:k]
        new_cids[-k:] = cids[-k:]
        score_threshold = 0.1
        M, T = landmarks.shape[:2]
        for t in range(k, T-k):
            left = np.array([landmarks[new_cids[_t], _t] for _t in range(t - k, t) if new_cids[_t] != -1 and landmarks_scores[new_cids[_t], _t].mean() > 0.2])
            right = np.array([landmarks[cids[_t], _t] for _t in range(t, t + k) if cids[_t] != -1])
            people = np.array([landmarks[p, t] for p in range(M) if landmarks_scores[p, t].mean() > score_threshold])
            if not people.any():
                continue
            if not left.any():
                left = np.zeros((0, 17, 2))
            if not right.any():
                right = np.zeros((0, 17, 2))
            neighbors = np.concatenate([left, right])
            if neighbors.any():
                dists = np.linalg.norm(neighbors[:, None] - people[None], axis=-1)
                for m in range(dists.shape[1]):
                    dists[:, m, :] = np.where(landmarks_scores[m, t] > score_threshold, dists[:, m, :], np.nan)
                dists = np.nanmean(dists, axis=2)#dists.mean(axis=2)
                majority = np.argmin(dists, axis=1)
                majority_votes = np.argmax(np.bincount(majority))
                if cids[t] != -1 and cids[t] != majority_votes:
                    print(f'Changing {cids[t]} to {majority_votes} at frame {t}')
                    new_cids[t] = majority_votes
                elif cids[t] == -1 and np.mean(dists[:, majority_votes]) < 180:
                    print(f'Assigning new cid: {majority_votes} at frame {t}')
                    new_cids[t] = majority_votes
                else:
                    new_cids[t] = cids[t]
            else:
                new_cids[t] = cids[t]
        return new_cids

class PyfeatDataExtractor(VisualizerDataExtractor):
    def __init__(self, graph_layout=PYFEAT_FACIAL, scale=1):
        super().__init__(graph_layout, scale)

    def _extract(self, cfg):
        cfg['pkl_path'] = osp.join(cfg['paths']['processed'], cfg['name'], 'barni', f'{cfg["name"]}.pkl')
        pkl_path = cfg['pkl_path']
        data = read_pkl(pkl_path)
        landmarks, landmarks_scores = data['landmarks'].astype(int) * self.scale, data['face_boxes'][:, :, 4]
        landmarks_scores = np.array([landmarks_scores] * 68).transpose((1, 2, 0))
        boxes = data['face_boxes'][:, :, :4].astype(int) * self.scale
        boxes = boxes.reshape(*boxes.shape[:-1], 2, 2)
        boxes[:, :, 0] += boxes[:, :, 1] // 2
        cids = data['child_ids'].astype(int)
        cids[cids == 255] = -1
        result = {'landmarks': landmarks, 'landmarks_scores': landmarks_scores, 'aus': data['aus'], 'emotions': data['emotions'],
                  'boxes': boxes, 'face_boxes': boxes, 'rotations': data['rotations'], 'child_ids': cids,
                  'resolution': (np.array(data['resolution']) * self.scale).astype(int) , 'fps': data['fps'], 'frame_count': data['frame_count'], 'duration_seconds': data['length'],
                  'video_path': data['video_path'], 'filename': data['filename'], 'feat_path': data['feat_path'], 'skip_frames': data['skip_frames'],
                  'au_cols': data['au_cols'], 'emotion_cols': data['emotion_cols'], 'rotation_cols': data['rotation_cols'], 'landmark_cols': data['landmark_cols'], 'face_cols': data['face_cols']}
        self._assign_labels(result, cids)
        return result