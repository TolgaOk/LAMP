import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import logging
from collections import defaultdict
from lamp.logger import DataLogger
from lamp.filters import DataLogFilter
import numpy as np


class MatplotHandler(logging.Handler):

    figure_ind = 0
    handler_counter = 0

    def __init__(self, capacity, flushOnClose=True):
        super().__init__()
        self.capacity = capacity
        self.buffer = list()
        self.flushOnClose = flushOnClose
        MatplotHandler.handler_counter += 1

    def emit(self, record):
        raise NotImplemented

    def flush(self):
        raise NotImplemented

    def flush_and_plot(self, *args):
        self.flush(*args)
        plt.show()

    def close(self):
        try:
            if self.flushOnClose:
                MatplotHandler.handler_counter -= 1

            if MatplotHandler.handler_counter == 0:
                plt.show()
        finally:
            self.acquire()
            try:
                logging.Handler.close(self)
            finally:
                self.release()


class MatplotScalarHandler(MatplotHandler):

    def __init__(self, capacity=None, flushOnClose=True, **kwargs):
        super().__init__(capacity, flushOnClose=flushOnClose)
        self.capacity = capacity
        self.plot_kwargs = kwargs
        self.buffer = defaultdict(lambda: defaultdict(list))
        self.addFilter(DataLogFilter(DataLogger.PlotType.SCALAR))

    def emit(self, record):
        env = record.plot_info["env"]
        trace = record.plot_info["trace"]
        self.buffer[env][trace].append(record.data)

    def flush(self):
        for env, traces in self.buffer.items():
            MatplotHandler.figure_ind += 1
            plt.figure(MatplotHandler.figure_ind)
            plt.title(env)
            for trace, data in traces.items():
                plt.plot(data, **self.plot_kwargs, label=trace)
            plt.legend()
        # TODO: Better legends.


class MatplotImageHandler(MatplotHandler):

    def __init__(self, capacity=None, flushOnClose=True, **kwargs):
        super().__init__(capacity, flushOnClose=flushOnClose)
        self.capacity = capacity
        self.plot_kwargs = kwargs
        self.addFilter(DataLogFilter(DataLogger.PlotType.IMAGE))

    def emit(self, record):
        self.buffer.append(record)

    def flush(self):
        for record in self.buffer:
            images = record.data
            nimg = nrow = ncol = 1
            if len(images.shape) == 4:
                nimg = images.shape[0]
                nrow = int(np.ceil(np.sqrt(nimg)))
                ncol = int(np.ceil(nimg/nrow))
                if images.shape[1] == 1:
                    images = images.squeeze(1)
            elif len(images.shape) in (3, 2):
                images = images[np.newaxis, ...]

            if len(images.shape) == 4:
                images = np.transpose(images, (0, 2, 3, 1))

            env = record.plot_info["env"]
            fig = plt.figure(MatplotHandler.figure_ind)
            plt.title(env)
            plt.box(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.gca().axes.get_xaxis().set_visible(False)
            grid = ImageGrid(fig, 111,
                             nrows_ncols=(nrow, ncol),
                             axes_pad=0.01,
                             )
            for i in range(nimg):
                try:
                    cmap = record.plot_info["cmap"].lower()
                except KeyError:
                    cmap = None
                MatplotHandler.figure_ind += 1
                print(images[i].shape)
                grid[i].imshow(images[i], cmap=cmap)
