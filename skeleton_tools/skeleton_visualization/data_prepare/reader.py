from abc import abstractmethod, ABC
from os import path as osp

import numpy as np
import cv2

class Reader(ABC):
    def __init__(self, scale):
        self.scale=scale
    @abstractmethod
    def get(self):
        pass

    def read(self):
        ret, frame = self.get()
        if ret:
            frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale)
        return ret, frame
    def release(self):
        pass

class VideoReader(Reader):
    def __init__(self, video_path, scale=1):
        super().__init__(scale)
        if not osp.exists(video_path):
            raise FileNotFoundError(f'Video "{video_path}" not found')
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

    def get(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

class DefaultReader(Reader):
    def __init__(self, resolution, scale=1, background_color=(255, 255, 255)):
        super().__init__(scale)
        self.background_color = np.array(background_color)
        self.n_channels = len(self.background_color)
        self.resolution = resolution
        self.background = (np.ones((*resolution[::-1], self.n_channels), dtype=np.uint8) * self.background_color).astype(np.uint8)

    def get(self):
        return True, self.background.copy()
