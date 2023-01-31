from abc import ABC, abstractmethod

import cv2
import numpy as np
from matplotlib import pyplot as plt


class BasePainter(ABC):
    @abstractmethod
    def _paint(self, *args, **kwargs):
        pass

    @abstractmethod
    def _get(self, data, frame_id, person_id):
        pass

    def __call__(self, frame, data, frame_id, person_id):
        return self._paint(frame.copy(), *self._get(data, frame_id, person_id))


class BoxPainter(BasePainter):
    def __init__(self):
        super().__init__()

    def _paint(self, frame, bbox, color):
        c, r = bbox
        r = r // 2
        cv2.rectangle(frame, tuple((c - r).astype(int)), tuple((c + r).astype(int)), color=color, thickness=1)
        return frame

    def _get(self, data, frame_id, person_id):
        return data['boxes'][person_id, frame_id], tuple(int(c) for c in data['colors'][person_id, frame_id])


class TextPainter(BasePainter, ABC):
    def _paint(self, frame, loc, text, color):
        cv2.putText(frame, text, loc.astype(np.int32), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        return frame


class LabelPainter(TextPainter):
    def _get(self, data, frame_id, person_id):
        return data['label_location'][person_id, frame_id], data['label_text'][person_id, frame_id], tuple(int(c) for c in data['colors'][person_id, frame_id])


class PersonIdentityPainter(TextPainter):
    def _get(self, data, frame_id, person_id):
        return data['label_location'][person_id, frame_id], data['person_id'][person_id, frame_id], tuple(int(c) for c in data['colors'][person_id, frame_id])


class BlurPainter(BasePainter):
    def _paint(self, frame, center, radius):
        c_mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.circle(c_mask, center, radius, 1, thickness=-1)
        mask = cv2.bitwise_and(frame, frame, mask=c_mask)
        img_mask = frame - mask
        blur = cv2.blur(frame, (50, 50))
        mask2 = cv2.bitwise_and(blur, blur, mask=c_mask)  # mask
        final_img = img_mask + mask2
        return final_img

    def _get(self, data, frame_id, person_id):
        c, r = data['face_boxes'][person_id, frame_id]
        return c, max(r.max(), 0) // 2


class SaliencyPainter(BasePainter):
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


class GraphPainter(BasePainter):
    def __init__(self, graph_layout, epsilon=1e-1, line_thickness=5):
        self.graph_layout = graph_layout
        self.epsilon = epsilon
        self.line_thickness = line_thickness

    def _paint(self, frame, landmarks, landmarks_scores, color):
        for (v1, v2) in self.graph_layout.pairs():
            if np.any(landmarks_scores < self.epsilon):
                continue
            cv2.line(frame, tuple(landmarks[v1]), tuple(landmarks[v2]), color, thickness=self.line_thickness, lineType=cv2.LINE_AA)
        return frame

    def _get(self, data, frame_id, person_id):
        return data['landmarks'][person_id, frame_id], data['landmarks_scores'][person_id, frame_id], tuple(int(c) for c in data['colors'][person_id, frame_id])
