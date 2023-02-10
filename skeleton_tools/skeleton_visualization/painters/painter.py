from abc import ABC, abstractmethod
from os import path as osp
from functools import cmp_to_key

import cv2
import numpy as np
from matplotlib import pyplot as plt

from skeleton_tools.skeleton_visualization.painters.paint_utils import blur_area


class BasePainter(ABC):
    @abstractmethod
    def _paint(self, *args, **kwargs):
        pass


class GlobalPainter(BasePainter):
    @abstractmethod
    def _get(self, data, frame_id):
        pass

    def __call__(self, frame, data, frame_id):
        return self._paint(frame.copy(), *self._get(data, frame_id))


class LocalPainter(BasePainter):

    @abstractmethod
    def _get(self, data, frame_id, person_id):
        pass

    def __call__(self, frame, data, frame_id, person_id):
        return self._paint(frame.copy(), *self._get(data, frame_id, person_id))


class BoxPainter(LocalPainter):
    def __init__(self):
        super().__init__()

    def _paint(self, frame, bbox, color):
        c, r = bbox
        r = r // 2
        cv2.rectangle(frame, tuple((c - r).astype(int)), tuple((c + r).astype(int)), color=color, thickness=1)
        return frame

    def _get(self, data, frame_id, person_id):
        return data['boxes'][person_id, frame_id], tuple(int(c) for c in data['colors'][person_id, frame_id])


class TextPainter(LocalPainter, ABC):
    def _paint(self, frame, loc, text, color):
        cv2.putText(frame, text, loc.astype(np.int32), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        return frame


class LabelPainter(TextPainter):
    def _get(self, data, frame_id, person_id):
        (cx, cy), (w, h) = data['facebox'][person_id, frame_id]
        loc = np.array([cx - w // 2, cy - h // 2])
        return loc, data['label_text'][person_id, frame_id], tuple(int(c) for c in data['colors'][person_id, frame_id])


class PersonIdentityPainter(TextPainter):
    def _get(self, data, frame_id, person_id):
        (cx, cy), (w, h) = data['facebox'][person_id, frame_id]
        loc = np.array([cx + w // 2 - 20, cy - h // 2])
        return loc, data['person_id'][person_id, frame_id], tuple(int(c) for c in data['colors'][person_id, frame_id])

class ScorePainter(TextPainter):
    def _get(self, data, frame_id, person_id):
        (cx, cy), (w, h) = data['facebox'][person_id, frame_id]
        loc = np.array([cx - w // 2, cy - h // 2])
        return loc, data['landmarks_scores'][person_id, frame_id].mean(), tuple(int(c) for c in data['colors'][person_id, frame_id])

class PersonBlurPainter(LocalPainter):
    def _paint(self, frame, center, radius):
        return blur_area(frame, center, radius)

    def _get(self, data, frame_id, person_id):
        c, r = data['face_boxes'][person_id, frame_id]
        return c, max(r.max(), 0) // 2

class SaliencyPainter(LocalPainter):
    def __init__(self, cmap='inferno'):
        super().__init__()
        self.cmap = plt.get_cmap(cmap)

    def _paint(self, frame, feature_map, saliency_map):
        intensity = saliency_map.mean(axis=1)
        for j, i in zip(feature_map, intensity):
            cv2.circle(img=frame, center=j, radius=0, color=np.array(self.cmap(i)) * 255, thickness=int(i ** 0.5 * 20))
        return frame

    def _get(self, data, frame_id, person_id):
        return data['landmarks'][person_id, frame_id], data['saliency_map'][frame_id]


class GraphPainter(LocalPainter):
    def __init__(self, graph_layout, epsilon=1e-1, line_thickness=5):
        self.graph_layout = graph_layout
        self.epsilon = epsilon
        self.line_thickness = line_thickness

    def _paint(self, frame, landmarks, landmarks_scores, color):
        for (v1, v2) in self.graph_layout.pairs():
            if np.any(landmarks_scores[[v1, v2]] < self.epsilon):
                continue
            cv2.line(frame, tuple(landmarks[v1]), tuple(landmarks[v2]), color, thickness=self.line_thickness, lineType=cv2.LINE_AA)
        return frame

    def _get(self, data, frame_id, person_id):
        return data['landmarks'][person_id, frame_id], data['landmarks_scores'][person_id, frame_id], tuple(int(c) for c in data['colors'][person_id, frame_id])

class BlurPainter(GlobalPainter):
    def __init__(self, face_boxes):
        self.boxes = np.zeros_like(face_boxes)

        last_boxes = np.zeros((face_boxes.shape[0], *face_boxes.shape[2:]))
        for f in range(face_boxes.shape[1]):
            frame_boxes = face_boxes[:, f].tolist()
            frame_boxes = sorted(frame_boxes, key=cmp_to_key(self.compare))
            for j, b in enumerate(frame_boxes):
                if np.sum(b) > 0:
                    last_boxes[j] = b
                self.boxes[j, f] = last_boxes[j]
        for f in reversed(range(face_boxes.shape[1])):
            frame_boxes = self.boxes[:, f]
            for j, b in enumerate(frame_boxes):
                if np.sum(b) == 0:
                    self.boxes[j, f] = last_boxes[j]
                else:
                    last_boxes[j] = b

    def compare(self, b1, b2):
        return b1[0][1] < b2[0][1] if b1[0][0] == b2[0][0] else b1[0][0] < b2[0][0]

    def _paint(self, frame, boxes):
        for (c, r) in boxes:
            r = max(r.max(), 0) // 2
            frame = blur_area(frame, c, r)
        return frame

    def _get(self, data, frame_id):
        return self.boxes[:, frame_id],
