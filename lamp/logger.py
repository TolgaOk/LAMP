import logging
import torch
import numpy as np
from collections import defaultdict

# [ ] TODO: Log closed windows from previous runs!
# [x] TODO: Auto increment index for scalar plot!
# [x] TODO: If default named environment exit and env argument is None, change default!
# [ ] TODO: Add more plotting arguments!
# [ ] TODO: Add default configuration with all plotting handlers!
# [ ] TODO: Save plots at shutdown if path is given!
# [ ] TODO: Add safe reliease and acquire statements!
# [ ] TODO: Make filters complete, check for all necessary attributes!

class DataLogger(logging.getLoggerClass()):
    def scalar(self, value, index=None, env="main", win=None, trace="trace-1", **kwargs):
        level = logging.INFO
        if self.isEnabledFor(level):
            plot_info = dict(env=env, win=win, trace=trace, index=index)
            self._log(level, "Scalar logged!", None, extra=dict(
                plot_info=plot_info,
                data=value,
                recordtype="scalar"
            ))

    def image(self, image, env="main", win=None, cmap="Viridis", **kwargs):
        level = logging.INFO
        if self.isEnabledFor(level):
            if not isinstance(image, (torch.Tensor, np.ndarray)):
                raise TypeError("Image type: {}, but torch tensor tensor or numpy array is expected!".format(type(image)))
            if len(image.shape) == 3:
                plot_info = dict(env=env, win=win)
                self._log(level, "3D Image logged!", None, extra=dict(
                    plot_info=plot_info,
                    data=image,
                    recordtype="image"
                ))
            elif len(image.shape) == 2:
                plot_info = dict(env=env, win=win, cmap=cmap)
                self._log(level, "2D Image logged!", None, extra=dict(
                    plot_info=plot_info,
                    data=image,
                    recordtype="image"
                ))                                                   
            elif len(image.shape) == 4:
                plot_info = dict(env=env, win=win, cmap=cmap)
                self._log(level, "Batch of images are logged!", None, extra=dict(
                    plot_info=plot_info,
                    data=image,
                    recordtype="image"
                ))
            else:
                raise ValueError("Image shape dim: {}D, But expected 2D, 3D or 4D!".format(len(image.shape)))

    def histogram(self, tensor, index=None, env="main", win=None, **kwargs):
        level = logging.INFO
        if self.isEnabledFor(level):
            if not isinstance(tensor, (torch.Tensor, np.ndarray)):
                raise TypeError("Tensor type: {}, but torch or numpy array is expected!".format(type(tensor)))
            plot_info = dict(env=env, win=win, index=index)
            self._log(level, "Histogram logged!", None, extra=dict(
                plot_info=plot_info,
                data=tensor,
                recordtype="histogram"
            ))
            
    def hyperparameters(self, params, env="main", win=None, **kwargs):
        level = logging.INFO
        if self.isEnabledFor(level):
            plot_info = dict(env=env, win=win)
            self._log(level, "Hyperparameters logged!", None, extra=dict(
                plot_info=plot_info,
                data=params,
                recordtype="hyperparameters")
            )

    def video(self, videofile, env="main", win=None, **kwargs):
        level = logging.INFO
        print(videofile)
        if self.isEnabledFor(level):
            plot_info = dict(env=env, win=win)
            self._log(level, "Video is logged", None, extra=dict(
                plot_info=plot_info,
                data=videofile,
                recordtype="video"
            ))

class PlotFilter(logging.Filter):

    def __init__(self, filtertype, name=""):
        super().__init__(name=name)
        self.filtertype = filtertype

    def filter(self, record):
        try:
            record.plot_info
            return record.recordtype == self.filtertype
        except AttributeError:
            return False

class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError( key )
        else:
            self[key] = self.default_factory(key)
            return self[key]