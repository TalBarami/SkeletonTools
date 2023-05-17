from abc import ABC, abstractmethod
from functools import cmp_to_key

import cv2
import numpy as np

from skeleton_tools.skeleton_visualization.paint_components.paint_utils import blur_area


class BasePainter(ABC):

    @abstractmethod
    def _paint(self, *args, **kwargs):
        pass


class GlobalPainter(BasePainter):
    def __init__(self, painter):
        self.painter = painter

    def __call__(self, frame, data, frame_id):
        return self._paint(frame, data, frame_id)

    def _paint(self, frame, data, frame_id):
        paint = np.zeros_like(frame)
        for m in range(data['landmarks'].shape[0]):
            if np.all(data['landmarks'][m, frame_id] == 0):
                continue
            paint = self.painter(paint, data, frame_id, m)
        mask = np.all(paint == 0, axis=2)
        blend = cv2.addWeighted(frame, 1 - self.painter.alpha, paint, self.painter.alpha, 0)
        blend[mask] = frame[mask]
        return blend


class LocalPainter(BasePainter):
    def __init__(self, alpha=1.0, color=None):
        self.alpha = alpha
        self.color = color

    @abstractmethod
    def _get(self, data, frame_id, person_id):
        pass

    def __call__(self, frame, data, frame_id, person_id):
        return self._paint(frame, *self._get(data, frame_id, person_id))

    def _get_color(self, data, frame_id, person_id):
        if type(self.color) == list:
            color = self.color[person_id % len(self.color)]
        elif type(self.color) == tuple:
            color = self.color
        else:
            color = tuple(int(c) for c in data['colors'][person_id, frame_id])[::-1]
        return color


class ScaleAbsPainter(GlobalPainter):
    def __init__(self, active=True, alpha=1, beta=1):
        super().__init__(self)
        self.active = active
        self.alpha = alpha
        self.beta = beta

    def switch(self, active=None):
        if active is None:
            self.active = not self.active
        else:
            self.active = active

    def _paint(self, frame, data, frame_id):
        if self.active:
            frame = cv2.convertScaleAbs(frame, alpha=self.alpha, beta=self.beta)
        return frame

class BlurPainter(GlobalPainter):
    def __init__(self, data, active=True):
        super().__init__(self)
        self.active = active
        self.data = data
        face_boxes = self.data['face_boxes']
        blur_boxes = np.zeros_like(face_boxes)
        self.data['blur_boxes'] = blur_boxes

        last_boxes = np.zeros((face_boxes.shape[0], *face_boxes.shape[2:]))
        for f in range(face_boxes.shape[1]):
            frame_boxes = face_boxes[:, f].tolist()
            frame_boxes = sorted(frame_boxes, key=cmp_to_key(self.compare))
            for j, b in enumerate(frame_boxes):
                if np.sum(b) > 0:
                    last_boxes[j] = b
                blur_boxes[j, f] = last_boxes[j]
        for f in reversed(range(face_boxes.shape[1])):
            frame_boxes = blur_boxes[:, f]
            for j, b in enumerate(frame_boxes):
                if np.sum(b) == 0:
                    blur_boxes[j, f] = last_boxes[j]
                else:
                    last_boxes[j] = b

    def switch(self, active=None):
        if active is None:
            self.active = not self.active
        else:
            self.active = active

    def compare(self, b1, b2):
        return b1[0][1] < b2[0][1] if b1[0][0] == b2[0][0] else b1[0][0] < b2[0][0]

    def _paint(self, frame, data, frame_id):
        if self.active:
            boxes = self.data['blur_boxes'][:, frame_id]
            for (c, r) in boxes:
                r = max(r.max(), 0) // 2
                frame = blur_area(frame, c, r)
        return frame

