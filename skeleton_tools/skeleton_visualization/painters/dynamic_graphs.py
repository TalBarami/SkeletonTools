from abc import ABC, abstractmethod

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
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

    def to_numpy(self, fig):
        fig.tight_layout()
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = cv2.cvtColor(frame.reshape(fig.canvas.get_width_height()[::-1] + (3,)), cv2.COLOR_RGB2BGR)
        plt.close(fig)
        return frame

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
        pad = np.zeros((self.radius, data.shape[1]))
        super().__init__(data=np.concatenate((pad, data, pad), axis=0), height=height, filters=filters, dpi=dpi)

    def plot(self, i):
        fig, ax = plt.subplots(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
        x = np.arange(i-self.radius, i+self.radius+1)
        y = self.data[i:i+2*self.radius+1]
        ax.plot(x, y)
        ax.set(title=self.title, xlabel=self.xlabel, xlim=(x.min(), x.max()), ylabel=self.ylabel, ylim=self.ylim)
        ax.axvline(x=i, color='r', linestyle='dotted')
        ax.grid()
        ax.legend(self.legend, loc='upper right')
        return self.to_numpy(fig)

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
        return self.to_numpy(fig)

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
        return self.to_numpy(fig)

if __name__ == '__main__':
    T = 10000000
    dp = DynamicPolar('test', np.random.rand(T, 10), ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'], 400)
    for i in range(T):
        dp(i)