from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from skeleton_tools.openpose_layouts.body import COCO_LAYOUT
from skeleton_tools.openpose_layouts.face import PYFEAT_FACIAL
from skeleton_tools.utils.skeleton_utils import bounding_box
from skeleton_tools.utils.tools import read_pkl, write_pkl


class VisualizerDataExtractor(ABC):
    def __init__(self, graph_layout, scale):
        self.graph_layout = graph_layout
        self.scale = scale
        self.epsilon = 1e-1

    @abstractmethod
    def _extract(self, *args, **kwargs):
        pass

    def __call__(self, *data):
        return self._extract(*data)

    def _assign_labels(self, result, cids):
        M, T = result['landmarks'].shape[:2]
        result['label_text'] = np.array([['Child' if cids[t] == i else 'Adult' for t in range(T)] for i in range(M)])
        result['colors'] = np.array([[(255, 0, 0) if cids[t] == i else (0, 0, 255) for t in range(T)] for i in range(M)])
        # result['label_text'] = np.array([['Child' if i == 0 else 'Adult' for t in range(T)] for i in range(M)])
        # result['colors'] = np.array([[(255, 0, 0) if i == 0 else (0, 0, 255) for t in range(T)] for i in range(M)])


class MMPoseDataExtractor(VisualizerDataExtractor):
    def __init__(self, graph_layout=COCO_LAYOUT, scale=1):
        super().__init__(graph_layout, scale)

    def _gen_boxes(self, landmarks, landmarks_scores):
        M, T = landmarks.shape[:2]
        boxes = np.array([[bounding_box(landmarks[i, t], landmarks_scores[i, t]) for t in range(T)] for i in range(M)])
        return boxes

    def _extract(self, data, scores=None):
        if type(data) == str:
            data = read_pkl(data)
        landmarks, landmarks_scores, cids = data['keypoint'] * self.scale, data['keypoint_score'], data['child_ids'].astype(int)
        T = landmarks.shape[1]
        if scores is not None:
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
                  'child_ids': cids, 'child_detection_scores': data['child_detected'], 'predictions': scores}

        face_joints = [k for k, v in self.graph_layout.joints().items() if any([s in v for s in self.graph_layout.face_joints()])]
        result['boxes'] = self._gen_boxes(landmarks, landmarks_scores).astype(int)
        result['face_boxes'] = self._gen_boxes(landmarks[:, :, face_joints], landmarks_scores[:, :, face_joints]).astype(int)
        result['child_ids'] *= 0
        self._assign_labels(result, cids)
        return result


class PyfeatDataExtractor(VisualizerDataExtractor):
    def __init__(self, graph_layout=PYFEAT_FACIAL, scale=1):
        super().__init__(graph_layout, scale)

    def _extract(self, data):
        if type(data) == str:
            data = read_pkl(data)
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