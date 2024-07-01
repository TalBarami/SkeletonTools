from abc import ABC

import cv2
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
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
    def __init__(self, graph_layout, epsilon=1e-1, line_thickness=3, alpha=1.0, color=None,
                 child_only=False, tracking=False, limbs=True, heatmap=False, show_joint_ids=False):
        super().__init__(alpha=alpha, color=color)
        self.graph_layout = graph_layout
        self.epsilon = epsilon
        self.line_thickness = line_thickness
        self.child_only = child_only
        self.tracking = tracking
        self.limbs = limbs
        self.heatmap = heatmap
        self.show_joint_ids = show_joint_ids
        self.cmap = (np.array(sns.color_palette('bwr', 512)) * 255).astype(np.uint8)
        self.red = np.flip(self.cmap[:256], axis=0)
        self.blue = self.cmap[256:]

    def _paint(self, frame: np.ndarray, landmarks: np.ndarray, landmarks_scores: np.ndarray, color: tuple, is_child: bool):
        if self.child_only and not is_child:
            return frame
        if self.limbs:
            for (v1, v2) in self.graph_layout.pairs():
                if np.any(landmarks_scores[[v1, v2]] < self.epsilon):
                    continue
                (x1, y1), (x2, y2) = landmarks[[v1, v2]].astype(np.int32)
                cv2.line(frame, (x1, y1), (x2, y2), color, thickness=self.line_thickness, lineType=cv2.LINE_AA)
        for v, (x, y) in enumerate(landmarks):
            if landmarks_scores[v] < self.epsilon:
                continue
            if self.heatmap:
                if is_child:
                    _color = self.red[int(np.clip(landmarks_scores[v], 0, 1) * 255)]
                else:
                    _color = self.blue[int(np.clip(landmarks_scores[v], 0, 1) * 255)]
                mask_img = np.zeros(frame.shape, dtype='uint8')
                cv2.circle(mask_img, (x, y), 10, _color.tolist(), -1)
                mask_img = (blur_area(mask_img, (x, y), 10, kernel_size=(10, 10)) * landmarks_scores[v]).astype(np.uint8)
                frame = np.where(mask_img > 0, mask_img, frame)
            else:
                cv2.circle(frame, (x, y), 0, color, thickness=-1)
            if self.show_joint_ids:
                cv2.putText(frame, str(v), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return frame

    def _get(self, data, frame_id, person_id):
        return data['landmarks'][person_id, frame_id], data['landmarks_scores'][person_id, frame_id], self._get_color(data, frame_id, person_id, col_name='person_colors' if self.tracking else 'colors'), data['child_ids'][frame_id] == person_id


class TextPainter(LocalPainter, ABC):
    def __init__(self, alpha=1.0, color=None, scale=1.0):
        super().__init__(alpha=alpha, color=color)
        self.scale = scale

    def _paint(self, frame, loc, text, color):
        if type(text) in [float, np.float32, np.float64, int, np.int32, np.int64]:
            text = f'{text:.3f}'
        cv2.putText(frame, text, loc.astype(np.int32), cv2.FONT_HERSHEY_SIMPLEX, 1.5 * self.scale, color, int(2 * self.scale), cv2.LINE_AA)
        return frame


class CustomTextPainter(TextPainter):
    def __init__(self, location, key, child_only=False):
        super().__init__()
        self.location = np.array(location)
        self.key = key
        self.child_only = child_only

    def _paint(self, frame, loc, text, color, is_child):
        if self.child_only and not is_child:
            return frame
        return super()._paint(frame, loc, text, color)

    def _get(self, data, frame_id, person_id):
        return self.location, str(data[self.key][person_id, frame_id]), self._get_color(data, frame_id, person_id), data['child_ids'][frame_id] == person_id


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
