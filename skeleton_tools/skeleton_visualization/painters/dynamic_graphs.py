from abc import ABC, abstractmethod

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import _pylab_helpers

plt.rcParams.update({'font.size': 10})

# class DynamicGraph(ABC):
#     def __init__(self, data, height, filters, dpi):
#         self._data = data.copy()
#         self.width, self.height = int(1.5 * height), height
#         self.filters = filters
#         self.dpi = dpi
#         for f in filters:
#             data = f(data)
#         self.data = data
#         nona = data[~np.isnan(data)]
#         self.ylim = (np.min(nona), np.max(nona))
#
#         self.fig, self.ax = plt.subplots(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
#         self.lines = self._get_lines()
#         if self.fig.stale:
#             self.fig.canvas.draw_idle()
#         self.fig.canvas.start_event_loop(0.1)
#         self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
#
#     def to_numpy(self, fig):
#         frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#         frame = cv2.cvtColor(frame.reshape(fig.canvas.get_width_height()[::-1] + (3,)), cv2.COLOR_RGB2BGR)
#         return frame
#
#     def __call__(self, i):
#         x, ys = self._plot(i)
#         self.fig.canvas.restore_region(self.bg)
#         for ln, y in zip(self.lines, ys.T):
#             ln.set_ydata(y)
#             self.ax.set_xlim(x[0], x[-1])
#             ln.set_xdata(x)
#             self.ax.draw_artist(ln)
#         frame = self.to_numpy(self.fig)
#         return frame
#
#     @abstractmethod
#     def _get_lines(self):
#         pass
#
#     @abstractmethod
#     def _plot(self, i):
#         pass
#
#
# class DynamicSignal(DynamicGraph):
#     def __init__(self, title, data, legend, xlabel, ylabel, window_size, height, filters=(), dpi=128):
#         self.title, self.legend, self.xlabel, self.ylabel = title, legend, xlabel, ylabel
#         self.window_size = window_size
#         self.radius = self.window_size // 2
#         pad = np.zeros((self.radius, data.shape[1]))
#         super().__init__(data=np.concatenate((pad, data, pad), axis=0), height=height, filters=filters, dpi=dpi)
#
#     def _get_lines(self):
#         ax = self.ax
#         x, y = self._plot(0)
#         lines = ax.plot(x, y, animated=True)
#         line = ax.axvline(x=0, color='r', linestyle='dotted')
#         ax.set(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel, ylim=self.ylim)
#         ax.grid()
#         ax.legend(self.legend, loc='upper right')
#         return lines
#
#     def _plot(self, i):
#         x = np.arange(i-self.radius, i+self.radius+1)
#         y = self.data[i:i+2*self.radius+1]
#         return x, y
#
# class DynamicBar(DynamicGraph):
#     def __init__(self, title, data, legend, xlabel, ylabel, height, filters=(), dpi=128):
#         super().__init__(data=data, height=height, filters=filters, dpi=dpi)
#         self.title, self.legend, self.xlabel, self.ylabel = title, legend, xlabel, ylabel
#
#     def _plot(self, i):
#         fig, ax = plt.subplots(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
#         x, y = np.arange(self.data.shape[1]), self.data[i]
#         ax.bar(x, y, 0.5)
#         ax.set(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel, ylim=self.ylim, xticks=x, xticklabels=self.legend)
#         ax.grid()
#         return self.to_numpy(fig)
#
# class DynamicPolar(DynamicGraph):
#     def __init__(self, title, data, legend, height, filters=(), dpi=128):
#         super().__init__(data=data, height=height, filters=filters, dpi=dpi)
#         self.title, self.legend = title, legend
#     def _plot(self, i):
#         fig, ax = plt.subplots(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi, subplot_kw=dict(polar=True))
#         x, y = np.linspace(0, 2 * np.pi, self.data.shape[1] + 1), self.data[i]
#         ax._plot(x, np.concatenate((y, [y[0]])))
#         ax.set(title=self.title, xticks=x, xticklabels=self.legend + [''], yticks=[], ylim=(0, 1))
#         ax.grid()
#         ax.fill_between(x, np.concatenate((y, [y[0]])), alpha=0.2)
#         return self.to_numpy(fig)
#

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
#
if __name__ == '__main__':
    manager = _pylab_helpers.Gcf.get_active()
    T = 10000000
    ds = DynamicSignal('test', np.random.rand(T, 2), ['a', 'b'], 'x', 'y', 400, 400)
    for i in range(T):
        frame = ds(i)
        cv2.imshow('test', frame)
        cv2.waitKey(1)