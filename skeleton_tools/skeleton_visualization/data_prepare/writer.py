from abc import abstractmethod, ABC
from os import path as osp

import cv2
import numpy as np
from skimage.color import rgba2rgb


class Writer(ABC):
    def __init__(self, out_path):
        self.out_path = out_path
        self.writer = None

    @abstractmethod
    def write(self, frame):
        pass

    def release(self):
        pass


class VideoWriter(Writer):
    def __init__(self, out_path, fps, resolution):
        _, ext = osp.splitext(out_path)
        super().__init__(out_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.out_path, fourcc, fps, resolution)

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()


class ImageWriter(Writer):
    def __init__(self, out_path, start=0):
        super().__init__(out_path)
        self.i = start

    def write(self, frame):
        if frame.shape[-1] == 4:
            top, bottom, left, right = [20] * 4
            frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
            frame[:, :, 3] = (255 * np.where((frame[:, :, :3] != 255).any(axis=2), 1, 0.6)).astype(np.uint8)
        cv2.imwrite(osp.join(self.out_path, f'{self.i}.jpg'), frame)
        self.i += 1
