from abc import ABC, abstractmethod

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import _pylab_helpers
from matplotlib.ticker import FuncFormatter
from scipy.signal import savgol_filter

from skeleton_tools.skeleton_visualization.paint_components.frame_painters.local_painters import GraphPainter
from skeleton_tools.skeleton_visualization.paint_components.paint_utils import fig2np

plt.rcParams.update({'font.size': 10})

class DynamicGraph(ABC):
    def __init__(self, data, height, width, filters, dpi):
        self.data = data
        self.width, self.height = int(1.5 * height) if width is None else width, height
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

def format_time(x, pos):
    x = x / 25
    minutes = int(x // 60)
    seconds = int(x % 60)
    return f'{minutes:02d}:{seconds:02d}'

class DynamicSignal(DynamicGraph):
    def __init__(self, title, data, legend, xlabel, ylabel, window_size, height, width=None, filters=(), dpi=128, xlabel_format=None):
        self.title, self.legend, self.xlabel, self.ylabel = title, legend, xlabel, ylabel
        self.window_size = window_size
        self.radius = self.window_size // 2
        self.ticks = 50
        self.xlabel_format = xlabel_format
        pad = np.zeros((self.radius, data.shape[1]))
        super().__init__(data=np.concatenate((pad, data, pad), axis=0), height=height, width=width, filters=filters, dpi=dpi)

    def plot(self, i):
        fig, ax = plt.subplots(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
        mn, mx = i - self.radius, i + self.radius + 1
        x = np.arange(mn, mx)
        y = self.data[i:i+2*self.radius+1]
        ax.plot(x, y)
        ax.set(title=self.title, xlabel=self.xlabel, xlim=(mn, mx), ylabel=self.ylabel, ylim=self.ylim)
        if self.xlabel_format == 'time':
            ax.xaxis.set_major_formatter(FuncFormatter(format_time))
        ax.axvline(x=i, color='r', linestyle='dotted')
        ax.grid()
        if self.legend:
            ax.legend(self.legend, loc='upper right')
        return fig2np(fig)

class DynamicBar(DynamicGraph):
    def __init__(self, title, data, legend, xlabel, ylabel, height, width=None, filters=(), dpi=128):
        super().__init__(data=data.copy(), height=height, width=width, filters=filters, dpi=dpi)
        self.title, self.legend, self.xlabel, self.ylabel = title, legend, xlabel, ylabel

    def plot(self, i):
        fig, ax = plt.subplots(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
        x, y = np.arange(self.data.shape[1]), self.data[i]
        ax.bar(x, y, 0.5)
        ax.set(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel, ylim=self.ylim, xticks=x, xticklabels=self.legend)
        ax.grid()
        return fig2np(fig)

def interpolate(x):
    nans, y = np.isnan(x), lambda z: z.nonzero()[0]
    x[nans] = np.interp(y(nans), y(~nans), x[~nans])
    x = savgol_filter(x, 31, 3, axis=0)
    if len(x.shape) == 1:
        x = np.expand_dims(x, 1)
    return x

class DynamicPolar(DynamicGraph):
    def __init__(self, title, data, legend, height, width=None, filters=(), dpi=128):
        super().__init__(data=interpolate(data), height=height, width=width, filters=filters, dpi=dpi)
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
    def __init__(self, data, height, layout, epsilon, child_only=False, width=None, tracking=False, limbs=True, heatmap=False):
        self.child_only = child_only
        self.painter = GraphPainter(layout, epsilon, child_only=self.child_only, limbs=limbs, tracking=tracking, heatmap=heatmap)
        self.width, self.height = int(1.5 * height) if width is None else width, height
        landmarks = data['landmarks'].copy()
        (w, h) = data['resolution']
        # self.scaled_width = int(self.width * w / h)
        # landmarks[:, :, :, 0] = landmarks[:, :, :, 0] * self.scaled_width / w
        landmarks[:, :, :, 0] = landmarks[:, :, :, 0] * self.width / w
        landmarks[:, :, :, 1] = landmarks[:, :, :, 1] * self.height / h
        self.m = landmarks.shape[0]
        self.data = {
            'landmarks': landmarks,
            'landmarks_scores': data['landmarks_scores'],
            'colors': data['colors'],
            'child_ids': data['child_ids'],
        }
        if 'person_ids' in data.keys():
            self.data['person_colors'] = data['person_colors']

    def plot(self, i):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # d = self.width - self.scaled_width
        for j in range(self.m):
            frame = self.painter(frame, self.data, i, j)
            # frame[:, d//2:d//2+self.scaled_width] = self.painter(frame[:, d//2:d//2+self.scaled_width], self.data, i, j)
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
