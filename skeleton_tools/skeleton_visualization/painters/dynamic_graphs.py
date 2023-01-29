from abc import ABC, abstractmethod

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

class DynamicGraph(ABC):
    def __init__(self, height, dpi):
        self.width, self.height = int(1.5 * height), height
        self.dpi = dpi

    def to_numpy(self, fig):
        fig.tight_layout()
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = cv2.cvtColor(frame.reshape(fig.canvas.get_width_height()[::-1] + (3,)), cv2.COLOR_RGB2BGR)
        return frame

    def __call__(self, i):
        return self.plot(i)

    @abstractmethod
    def plot(self, i):
        pass

class DynamicSignal(DynamicGraph):
    def __init__(self, title, signals, legend, xlabel, ylabel, window_size, height, dpi=128):
        super().__init__(height=height, dpi=dpi)
        self.title, self.legend, self.xlabel, self.ylabel = title, legend, xlabel, ylabel
        self.window_size = window_size
        self.radius = self.window_size // 2
        pad = np.zeros((self.radius, signals.shape[1]))
        self.signals = np.concatenate((pad, signals, pad), axis=0)
        self.ylim = (np.min(self.signals), np.max(self.signals))

    def plot(self, i):
        fig, ax = plt.subplots(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
        x = np.arange(i-self.radius, i+self.radius+1)
        y = self.signals[i:i+2*self.radius+1]
        ax.plot(x, y)
        ax.set(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel, ylim=self.ylim)
        ax.axvline(x=i, color='r', linestyle='dotted')
        ax.grid()
        ax.legend(self.legend, loc='upper right')
        return self.to_numpy(fig)

class DynamicBar(DynamicGraph):
    def __init__(self, title, data, legend, xlabel, ylabel, height, dpi=128):
        super().__init__(height=height, dpi=dpi)
        self.title, self.legend, self.xlabel, self.ylabel = title, legend, xlabel, ylabel
        self.data = data
        self.ylim = (np.min(self.data), np.max(self.data))

    def plot(self, i):
        fig, ax = plt.subplots(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
        x, y = np.arange(self.data.shape[1]), self.data[i]
        ax.bar(x, y, 0.5)
        ax.set(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel, ylim=self.ylim, xticks=x, xticklabels=self.legend)
        ax.grid()
        return self.to_numpy(fig)

class DynamicPolar(DynamicGraph):
    def __init__(self, title, data, legend, height, dpi=128):
        super().__init__(height=height, dpi=dpi)
        self.title, self.legend = title, legend
        self.data = data

    def plot(self, i):
        fig, ax = plt.subplots(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi, subplot_kw=dict(polar=True))
        x, y = np.linspace(0, 2 * np.pi, self.data.shape[1]), self.data[i]
        ax.plot(np.concatenate((x, [x[0]])), np.concatenate((y, [y[0]])))
        ax.set(title=self.title, xticks=x, xticklabels=self.legend, yticks=[], ylim=(0, 1))
        ax.grid()
        ax.fill_between(x, y, alpha=0.2)
        return self.to_numpy(fig)
