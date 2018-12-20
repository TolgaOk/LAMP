import logging
import torch
import numpy as np
import visdom
from collections import defaultdict 
from .logger import PlotFilter
from .logger import keydefaultdict

class VisdomHandlers(logging.Handler):

    def __init__(self, level=logging.NOTSET, overwrite_window=True, ip_addr="localhost", port=8097):
        super().__init__(level=level)
        # self.log = logging.getLogger(__name__+"."+__class__.__name__)
        self.ip_addr = ip_addr
        self.port = port
        self.overwrite_window = overwrite_window
        self.is_win_used = defaultdict(lambda: False)
 
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
        server = "http://"+str(self.ip_addr)
        self.viz = visdom.Visdom(server=server, port=self.port)

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

class VisdomScalarHandler(VisdomHandlers):

    def __init__(self, level=logging.NOTSET, overwrite_window=True, ip_addr="localhost", port=8097, lock=None):
        super().__init__(level=level, overwrite_window=overwrite_window, ip_addr=ip_addr, port=port)
        self.addFilter(PlotFilter("scalar"))
        self._default_win_name = keydefaultdict(
                            lambda env: self.get_available_name("scalar-1", env)
                        )
        self.torchlock = lock
        if self.torchlock:
            self.manager = torch.multiprocessing.Manager()
            self._last_indexes = self.manager.dict()
        else:
            self._last_indexes = defaultdict(lambda: -1)

    def get_index(self, state):
        if self.torchlock:
            try:
                return self._last_indexes[state].value
            except KeyError as e:
                self._last_indexes[state] = self.manager.Value("i", -1)
                return self._last_indexes[state].value
        else:
            return self._last_indexes[state]

    def inc_index(self, state):
        if self.torchlock:
            self._last_indexes[state].value += 1
        else:
            self._last_indexes[state] += 1

    def get_default_win_name(self, env):
        return self._default_win_name[env]
        
    def _emit(self, record):
        plot_info = record.plot_info
        value = record.data
        win=plot_info["win"]
        env=plot_info["env"]
        trace = plot_info["trace"]
        x = plot_info["index"]
        y = value
        opts = dict(
            title = win
        )
        if x is None:
            x = self.get_index((env, win, trace)) + 1
        self.inc_index((env, win, trace))
        self.viz.line(Y=[y], X=[x], win=win ,env=env, update="append", name=trace, opts=opts)

class VisdomImageHandler(VisdomHandlers):

    def __init__(self, level=logging.NOTSET, overwrite_window=True, ip_addr="localhost", port=8097):
        super().__init__(level=level, overwrite_window=overwrite_window, ip_addr=ip_addr, port=port)
        self.addFilter(PlotFilter("image"))
        self._default_win_name = keydefaultdict(
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

            grid = np.zeros((nchan, height*nrow, width*ncol), dtype=image.dtype)
            for i in range(nrow):
                for j in range(ncol):
                    img_indx = i*ncol + j 
                    if img_indx >= nimg:
                        continue
                    grid[i*height:(i+1)*height, j*width:(j+1)*width] = image[img_indx]
            image = grid.squeeze(0)

        if len(image.shape) == 2:
            heigth, width = image.shape
            msg = dict(
                data = [
                    dict(
                        z = image.tolist(),
                        type = "heatmap",
                        colorscale = plot_info["cmap"]
                    )
                ],
                layout = dict(
                    yaxis = dict(
                        autoarange = "reversed"
                    ),
                    width = width,
                    height = height
                )
            )
            self.viz._send(msg)

        elif len(image.shape) == 3:
            self.viz.image(image.numpy())

class VisdomHistHandler(VisdomHandlers):

    def __init__(self, level=logging.NOTSET, overwrite_window=True, ip_addr="localhost", port=8097):
        super().__init__(level=level, overwrite_window=overwrite_window, ip_addr=ip_addr, port=port)
        self.addFilter(PlotFilter("historgram"))
        self._default_win_name = keydefaultdict(
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
            title = plot_info["win"]
        )
        self.viz.histogram(X=value, win=plot_info["win"], env=plot_info["env"], opts=opts)

class VisdomParameterHandler(VisdomHandlers):

    def __init__(self, level=logging.NOTSET, overwrite_window=True, ip_addr="localhost", port=8097):
        super().__init__(level=level, overwrite_window=overwrite_window, ip_addr=ip_addr, port=port)
        self.addFilter(PlotFilter("hyperparameters"))
        self._default_win_name = keydefaultdict(
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
            params_str += (  "<p>"
                                +"<pre>"
                                    +"<strong>"
                                        +str(key)
                                    +"</strong>"
                                    +"  :  "
                                    +str(value)
                                +"</pre>"
                            +"</p>")
        self.viz.text(text=params_str, win=plot_info["win"], env=plot_info["env"]) 

class VisdomVideoHandler(VisdomHandlers):
    def __init__(self, level=logging.NOTSET, overwrite_window=True, ip_addr="localhost", port=8097):
        super().__init__(level=level, overwrite_window=overwrite_window, ip_addr=ip_addr, port=port)
        self.addFilter(PlotFilter("video"))
        self._default_win_name = keydefaultdict(
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