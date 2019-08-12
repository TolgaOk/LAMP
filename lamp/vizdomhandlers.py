""" Logging Handlers for visdom
"""
import logging
import torch
import numpy as np
import visdom
from collections import defaultdict
import socket

import logging.handlers as handlers
from lamp.logger import DataLogger
from lamp.filters import DataLogFilter
from lamp.utils import KeyDefaultdict
from lamp.utils import SharedDefaultDict
from lamp.utils import SharedKeyDefaultDict


class VisdomHandler(logging.Handler):

    def __init__(self, level, overwrite_window, ip_addr, port, manager=None):
        super().__init__(level=level)
        if manager:
            self.manager = manager
            self.is_win_used = SharedDefaultDict(manager, lambda: False)
        else:
            self.is_win_used = defaultdict(lambda: False)
        self.ip_addr = ip_addr
        self.port = port
        self.overwrite_window = overwrite_window

    def emit(self, record):
        try:
            self.viz
        except AttributeError:
            self.connect()

        env = record.plot_info["env"] or "main"
        win = record.plot_info["win"] or self.get_default_win_name(env)
        record.plot_info["win"] = win

        if not self.is_win_used[(env, win)]:
            if self.overwrite_window:
                self.viz.close(win, env)
            self.is_win_used[(env, win)] = True

        self._emit(record)

    def connect(self):
        if self._check_server():
            print("Server is active, connecting...!", flush=True)
            server = "http://"+str(self.ip_addr)
            self.viz = visdom.Visdom(server=server, port=self.port)
        else:
            print("Server is not active, no connection!", flush=True)

    def _check_server(self):
        # This function is taken from the comment of Roulbac to the issue below
        # https://github.com/facebookresearch/visdom/issues/156
        # TODO: Test the function with ip address as hostname
        is_used = False
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((self.ip_addr, self.port))
        except socket.error:
            is_used = True
        finally:
            s.close()
        return is_used

    def get_available_name(self, name, env):
        if self.overwrite_window:
            return name
        prefix, suffix = name.split("-")
        index = int(suffix)
        while index < 1000:
            win_name = prefix + "-" + str(index)
            if not self.viz.win_exists(win_name, env):
                return win_name
            index += 1
        # Warning All slots are full

    def increment_win_name(self, name, env):
        prefix, suffix = name.split("-")
        if self.overwrite_window:
            win_name = prefix + "-" + str(int(suffix)+1)
            return win_name
        else:
            index = int(suffix)+1
            while index < 1000:
                win_name = prefix + "-" + str(index)
                if not self.viz.win_exists(win_name, env):
                    return win_name
                index += 1

    def get_default_win_name(self, env):
        raise NotImplementedError

    def _emit(self, record):
        raise NotImplementedError


class VisdomScalarHandler(VisdomHandler):

    def __init__(self, level=logging.NOTSET, overwrite_window=True,
                 ip_addr="localhost", port=8097, manager=None):
        super().__init__(level=level, overwrite_window=overwrite_window,
                         ip_addr=ip_addr, port=port, manager=manager)
        self.addFilter(DataLogFilter(DataLogger.PlotType.SCALAR))
        if manager:
            self._default_win_name = \
                SharedKeyDefaultDict(manager,
                                     lambda env: self.get_available_name(
                                         "scalar-1", env)
                                     )
            self._last_indexes = SharedDefaultDict(manager, lambda: -1)
        else:
            self._default_win_name = KeyDefaultdict(
                lambda env: self.get_available_name("scalar-1", env)
            )
            self._last_indexes = defaultdict(lambda: -1)

    def _index(self, state):
        self._last_indexes[state] += 1
        return self._last_indexes[state]

    def get_default_win_name(self, env):
        return self._default_win_name[env]

    def _emit(self, record):
        plot_info = record.plot_info
        value = record.data
        win = plot_info["win"]
        env = plot_info["env"]
        trace = plot_info["trace"]
        x = plot_info["index"]
        y = value
        opts = dict(
            title=win,
            marginleft=30,
            marginright=30,
            marginbottom=80,
            margintop=30,
            width=300,
            height=300,
            dash="dot",
            # xtickmin=-6,
            # xtickmax=6,
            xlabel='Arbitrary',
            fillarea=False,
            # xtickvals=[0, 0.75, 1.6, 2],
            # ytickmin=0,
            # ytickmax=0.5,
            # ytickstep=0.5,
            # ztickmin=0,
            # ztickmax=1,
            # ztickstep=0.5,

        )
        if x is None:
            x = self._index((env, win, trace))
        self.viz.line(Y=[y], X=[x], win=win, env=env,
                      update="append", name=trace, opts=opts)


class VisdomImageHandler(VisdomHandler):

    def __init__(self, level=logging.NOTSET, overwrite_window=True,
                 ip_addr="localhost", port=8097):
        super().__init__(level=level, overwrite_window=overwrite_window,
                         ip_addr=ip_addr, port=port)
        self.addFilter(DataLogFilter(DataLogger.PlotType.IMAGE))
        self._default_win_name = KeyDefaultdict(
            lambda env: self.get_available_name("image-1", env)
        )

    def get_default_win_name(self, env):
        default_name = self._default_win_name[env]
        name = self.increment_win_name(default_name, env)
        self._default_win_name[env] = name
        return default_name

    def _emit(self, record):
        plot_info = record.plot_info
        image = record.data
        if isinstance(image, torch.tensor):
            image = image.numpy()

        if len(image.shape) == 4:
            nimg, nchan, height, width = image.shape
            nrow = int(np.ceil(np.sqrt(nimg)))
            ncol = int(np.ceil(nimg/nrow))

            grid = np.zeros((nchan, height*nrow, width*ncol),
                            dtype=image.dtype)
            for i in range(nrow):
                for j in range(ncol):
                    img_indx = i*ncol + j
                    if img_indx >= nimg:
                        continue
                    grid[i*height:(i+1)*height,
                         j*width:(j+1) * width] = image[img_indx]
            image = grid.squeeze(0)

        if len(image.shape) == 2:
            heigth, width = image.shape
            msg = dict(
                data=[
                    dict(
                        z=image.tolist(),
                        type="heatmap",
                        colorscale=plot_info["cmap"]
                    )
                ],
                layout=dict(
                    yaxis=dict(
                        autoarange="reversed"
                    ),
                    width=width,
                    height=height
                )
            )
            self.viz._send(msg)

        elif len(image.shape) == 3:
            self.viz.image(image.numpy())


class VisdomHistHandler(VisdomHandler):

    def __init__(self, level=logging.NOTSET, overwrite_window=True,
                 ip_addr="localhost", port=8097):
        super().__init__(level=level, overwrite_window=overwrite_window,
                         ip_addr=ip_addr, port=port)
        self.addFilter(DataLogFilter(DataLogger.PlotType.HISTOGRAM))
        self._default_win_name = KeyDefaultdict(
            lambda env: self.get_available_name("histogram-1", env)
        )

    def get_default_win_name(self, env):
        default_name = self._default_win_name[env]
        name = self.increment_win_name(default_name, env)
        self._default_win_name[env] = name
        return default_name

    def _emit(self, record):
        plot_info = record.plot_info

        value = record.data.reshape(-1)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        opts = dict(
            title=plot_info["win"]
        )
        self.viz.histogram(
            X=value, win=plot_info["win"], env=plot_info["env"], opts=opts)


class VisdomParameterHandler(VisdomHandler):

    def __init__(self, level=logging.NOTSET, overwrite_window=True,
                 ip_addr="localhost", port=8097):
        super().__init__(level=level, overwrite_window=overwrite_window,
                         ip_addr=ip_addr, port=port)
        self.addFilter(DataLogFilter(DataLogger.PlotType.HYPERPARAM))
        self._default_win_name = KeyDefaultdict(
            lambda env: self.get_available_name("hyperparam-1", env)
        )

    def get_default_win_name(self, env):
        default_name = self._default_win_name[env]
        name = self.increment_win_name(default_name, env)
        self._default_win_name[env] = name
        return default_name

    def _emit(self, record):
        plot_info = record.plot_info
        params = record.data
        params_str = "<h4>Hyperparameters</h4>"
        for key, value in params.items():
            params_str += ("<p>"
                           + "<pre>"
                           + "<strong>"
                           + str(key)
                           + "</strong>"
                           + "  :  "
                           + str(value)
                           + "</pre>"
                           + "</p>")
        self.viz.text(text=params_str,
                      win=plot_info["win"], env=plot_info["env"])


class VisdomVideoHandler(VisdomHandler):
    def __init__(self, level=logging.NOTSET, overwrite_window=True,
                 ip_addr="localhost", port=8097):
        super().__init__(level=level, overwrite_window=overwrite_window,
                         ip_addr=ip_addr, port=port)
        self.addFilter(DataLogFilter(DataLogger.PlotType.VIDEO))
        self._default_win_name = KeyDefaultdict(
            lambda env: self.get_available_name("video-1", env)
        )

    def get_default_win_name(self, env):
        default_name = self._default_win_name[env]
        name = self.increment_win_name(default_name, env)
        self._default_win_name[env] = name
        return default_name

    def _emit(self, record):
        plot_info = record.plot_info
        videofile = record.data
        self.viz.video(videofile=videofile)
