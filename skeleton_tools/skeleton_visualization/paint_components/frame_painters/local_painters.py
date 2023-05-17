from abc import ABC

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skeleton_tools.skeleton_visualization.paint_components.frame_painters.base_painters import LocalPainter

from skeleton_tools.skeleton_visualization.paint_components.paint_utils import blur_area


class BoxPainter(LocalPainter):
    def __init__(self, alpha=1.0, color=None):
        super().__init__(alpha=alpha, color=color)

    def _paint(self, frame, bbox, color):
        c, r = bbox
        r = r // 2
        cv2.rectangle(frame, tuple((c - r).astype(int)), tuple((c + r).astype(int)), color=color, thickness=1)
        return frame

    def _get(self, data, frame_id, person_id):
        return data['boxes'][person_id, frame_id], self._get_color(data, frame_id, person_id)


class GraphPainter(LocalPainter):
    def __init__(self, graph_layout, epsilon=1e-1, line_thickness=3, alpha=1.0, color=None, child_only=False):
        super().__init__(alpha=alpha, color=color)
        self.graph_layout = graph_layout
        self.epsilon = epsilon
        self.line_thickness = line_thickness
        self.child_only = child_only

    def _paint(self, frame: np.ndarray, landmarks: np.ndarray, landmarks_scores: np.ndarray, color: tuple, is_child: bool):
        if self.child_only and not is_child:
            return frame
        for (v1, v2) in self.graph_layout.pairs():
            if np.any(landmarks_scores[[v1, v2]] < self.epsilon):
                continue
            cv2.line(frame, tuple(landmarks[v1]), tuple(landmarks[v2]), color, thickness=self.line_thickness, lineType=cv2.LINE_AA)
        for v, (x, y) in enumerate(landmarks):
            if landmarks_scores[v] < self.epsilon:
                continue
            cv2.circle(frame, (x, y), 0, color, thickness=-1)
        return frame

    def _get(self, data, frame_id, person_id):
        return data['landmarks'][person_id, frame_id], data['landmarks_scores'][person_id, frame_id], self._get_color(data, frame_id, person_id), data['child_ids'][frame_id] == person_id


class TextPainter(LocalPainter, ABC):
    def __init__(self, alpha=1.0, color=None, scale=1.0):
        super().__init__(alpha=alpha, color=color)
        self.scale = scale

    def _paint(self, frame, loc, text, color):
        if type(text) in [float, np.float32, np.float64, int, np.int32, np.int64]:
            text = f'{text:.3f}'
        cv2.putText(frame, text, loc.astype(np.int32), cv2.FONT_HERSHEY_SIMPLEX, 1.5 * self.scale, color, int(2 * self.scale), cv2.LINE_AA)
        return frame


class LabelPainter(TextPainter):
    def _get(self, data, frame_id, person_id):
        (cx, cy), (w, h) = data['face_boxes'][person_id, frame_id]
        loc = np.array([cx - w // 2, cy - h // 2])
        return loc, data['label_text'][person_id, frame_id], self._get_color(data, frame_id, person_id)


class PersonIdentityPainter(TextPainter):
    def __init__(self, color=None):
        super().__init__(color=color)
        self.offset = lambda: int(120 * self.scale)

    def _get(self, data, frame_id, person_id):
        (cx, cy), (w, h) = data['face_boxes'][person_id, frame_id]
        loc = np.array([cx + w // 2 + self.offset(), cy - h // 2])
        return loc, data['person_id'][person_id, frame_id], self._get_color(data, frame_id, person_id)


class ScorePainter(TextPainter):
    def __init__(self, color=None):
        super().__init__(color=color)
        self.offset = lambda: int(120 * self.scale)

    def _get(self, data, frame_id, person_id):
        (cx, cy), (w, h) = data['face_boxes'][person_id, frame_id]
        loc = np.array([cx - w // 2 + self.offset(), cy - h // 2])
        return loc, data['landmarks_scores'][person_id, frame_id].mean(), self._get_color(data, frame_id, person_id)


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
