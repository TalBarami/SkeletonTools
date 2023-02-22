from abc import ABC, abstractmethod

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import _pylab_helpers

from skeleton_tools.skeleton_visualization.painters.local_painters import GraphPainter
from skeleton_tools.skeleton_visualization.painters.paint_utils import fig2np

plt.rcParams.update({'font.size': 10})

class DynamicGraph(ABC):
    def __init__(self, data, height, filters, dpi):
        self._data = data.copy()
        self.width, self.height = int(1.5 * height), height
        self.filters = filters
        self.dpi = dpi
        for f in filters:
            data = f(data)
        self.data = data
        nona = data[~np.isnan(data)]
        self.ylim = (np.min(nona), np.max(nona))

    def __call__(self, i):
        return self.plot(i)

    @abstractmethod
    def plot(self, i):
        pass

class DynamicSignal(DynamicGraph):
    def __init__(self, title, data, legend, xlabel, ylabel, window_size, height, filters=(), dpi=128):
        self.title, self.legend, self.xlabel, self.ylabel = title, legend, xlabel, ylabel
        self.window_size = window_size
        self.radius = self.window_size // 2
        self.ticks = 50
        pad = np.zeros((self.radius, data.shape[1]))
        super().__init__(data=np.concatenate((pad, data, pad), axis=0), height=height, filters=filters, dpi=dpi)

    def plot(self, i):
        fig, ax = plt.subplots(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
        x = np.arange(i-self.radius, i+self.radius+1)
        y = self.data[i:i+2*self.radius+1]
        ax.plot(x, y)
        mn, mx = x.min(), x.max()
        # xticks = np.linspace(*(np.round(np.array([mn, mx])/self.ticks) * self.ticks), (mx - mn) // self.ticks + 1)
        ax.set(title=self.title, xlabel=self.xlabel, xlim=(mn, mx), ylabel=self.ylabel, ylim=self.ylim)
        ax.axvline(x=i, color='r', linestyle='dotted')
        ax.grid()
        ax.legend(self.legend, loc='upper right')
        return fig2np(fig)

class DynamicBar(DynamicGraph):
    def __init__(self, title, data, legend, xlabel, ylabel, height, filters=(), dpi=128):
        super().__init__(data=data, height=height, filters=filters, dpi=dpi)
        self.title, self.legend, self.xlabel, self.ylabel = title, legend, xlabel, ylabel

    def plot(self, i):
        fig, ax = plt.subplots(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
        x, y = np.arange(self.data.shape[1]), self.data[i]
        ax.bar(x, y, 0.5)
        ax.set(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel, ylim=self.ylim, xticks=x, xticklabels=self.legend)
        ax.grid()
        return fig2np(fig)

class DynamicPolar(DynamicGraph):
    def __init__(self, title, data, legend, height, filters=(), dpi=128):
        super().__init__(data=data, height=height, filters=filters, dpi=dpi)
        self.title, self.legend = title, legend
    def plot(self, i):
        fig, ax = plt.subplots(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi, subplot_kw=dict(polar=True))
        x, y = np.linspace(0, 2 * np.pi, self.data.shape[1] + 1), self.data[i]
        ax.plot(x, np.concatenate((y, [y[0]])))
        ax.set(title=self.title, xticks=x, xticklabels=self.legend + [''], yticks=[], ylim=(0, 1))
        ax.grid()
        ax.fill_between(x, np.concatenate((y, [y[0]])), alpha=0.2)
        return fig2np(fig)

class DynamicSkeleton:
    def __init__(self, data, height, layout, epsilon):
        self.painter = GraphPainter(layout, epsilon)
        self.width, self.height = int(1.5 * height), height
        landmarks = data['landmarks'].copy()
        (w, h) = data['resolution']
        landmarks[:, :, :, 0] = landmarks[:, :, :, 0] * self.width / w
        landmarks[:, :, :, 1] = landmarks[:, :, :, 1] * self.height / h
        self.m = landmarks.shape[0]
        self.data = {
            'landmarks': landmarks,
            'landmarks_scores': data['landmarks_scores'],
            'colors': data['colors'],
        }

    def plot(self, i):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for j in range(self.m):
            frame = self.painter(frame, self.data, i, j)
        return frame

    def __call__(self, i):
        return self.plot(i)

    # painter(paint_frame, video_data, i, j)

if __name__ == '__main__':
    manager = _pylab_helpers.Gcf.get_active()
    T = 10000000
    ds = DynamicSignal('test', np.random.rand(T, 2), ['a', 'b'], 'x', 'y', 400, 400)
    for i in range(T):
        frame = ds(i)
        cv2.imshow('test', frame)
        cv2.waitKey(1)
