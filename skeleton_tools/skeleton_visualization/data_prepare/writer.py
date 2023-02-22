from abc import abstractmethod, ABC
from os import path as osp

import cv2
from skimage.color import rgba2rgb


class Writer(ABC):
    def __init__(self, out_path):
        self.out_path = out_path
        self.writer = None

    @abstractmethod
    def write(self, frame):
        pass

    @abstractmethod
    def release(self):
        pass


class VideoWriter(Writer):
    def __init__(self, out_path, fps, resolution):
        super().__init__(out_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(out_path, fourcc, fps, resolution)

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()


class ImageWriter(Writer):
    def __init__(self, out_path, start=0):
        super().__init__(out_path)
        self.i = start

    def write(self, frame):
        cv2.imwrite(osp.join(self.out_path, f'{self.i}.jpg'), frame)
        self.i += 1

    def release(self):
        pass
