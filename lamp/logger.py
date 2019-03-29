"""
Logger class for additional data logging.
    
DataLogger extends the default logger class by including 
some data logging options such as: scalar, image, histogram,
hyperparameter and video.
"""

import logging
import torch
import numpy as np
from collections import defaultdict

# [ ] TODO: Log closed windows from previous runs!
# [x] TODO: Auto increment index for scalar plot!
# [x] TODO: If default named environment exit and env argument is None, change
#  default!
# [ ] TODO: Add more plotting arguments!
# [ ] TODO: Add default configuration with all plotting handlers!
# [ ] TODO: Save plots at shutdown if path is given!
# [ ] TODO: Add safe reliease and acquire statements!
# [ ] TODO: Make filters complete, check for all necessary attributes!


class DataLogger(logging.getLoggerClass()):
    """
    Support additional logging methods over base logging class.

    These methods includes: scalar, image, histogram, hyperparameter
    and video. Each method logs information about the plotting
    either on Visdom or Matplotlib.
    """

    def scalar(self, value, index=None, env="main",
               win=None, trace="trace-1", **kwargs):
        """
        Log scalar values for both online and offline plotting.

        Arguments
            value: Value to be logged.
            index: Coordinate of the value at the x axis (default None)
            env: Name of the environment for Visdom (default "main")
            win: Name of the window (default None)
            trace: Trace name of the value. There can be multiple
                traces on the plot. (default "trace-1")
        """
        level = logging.INFO
        if self.isEnabledFor(level):
            plot_info = dict(env=env, win=win, trace=trace, index=index)
            self._log(level, "Scalar logged!", None,
                      extra=dict(
                          plot_info=plot_info,
                          data=value,
                          recordtype="scalar")
                      )

    def image(self, image, env="main", win=None, cmap="Viridis", **kwargs):
        """
        Log images of different dimensions such as 2D, 3D or 4D.

        3D images are considered as RGB images. For 2D images heatmap
        is used to visualize while 4D ones are considered batch of RPG
        images.

        Arguments
            image: image array or tensor to be logged.
            env: Name of the environment for Visdom (default "main")
            win: Name of the window (default None)
            cmap: Colormap of the 2D images (default "Viridis")
        """
        level = logging.INFO
        if self.isEnabledFor(level):
            if not isinstance(image, (torch.Tensor, np.ndarray)):
                raise TypeError(
                    """Image type: {}, but torch tensor tensor 
                    or numpy array is expected!""".format(type(image))
                )
            if len(image.shape) == 3:
                plot_info = dict(env=env, win=win)
                self._log(level, "3D Image logged!", None,
                          extra=dict(
                              plot_info=plot_info,
                              data=image,
                              recordtype="image")
                          )
            elif len(image.shape) == 2:
                plot_info = dict(env=env, win=win, cmap=cmap)
                self._log(level, "2D Image logged!", None,
                          extra=dict(
                              plot_info=plot_info,
                              data=image,
                              recordtype="image"
                          )
                          )
            elif len(image.shape) == 4:
                plot_info = dict(env=env, win=win, cmap=cmap)
                self._log(level, "Batch of images are logged!", None,
                          extra=dict(
                              plot_info=plot_info,
                              data=image,
                              recordtype="image")
                          )
            else:
                raise ValueError(
                    """Image shape dim: {}D,
                    But expected 2D, 3D or 4D!""".format(len(image.shape))
                )

    def histogram(self, array, env="main", win=None, **kwargs):
        """
        Log multiple values for histogram plotting.

        Arguments
            array: Array or tensor of multiple values.
            env: Name of the environment for Visdom (default "main")
            win: Name of the window (default None)
        """
        level = logging.INFO
        if self.isEnabledFor(level):
            if not isinstance(array, (torch.Tensor, np.ndarray)):
                raise TypeError(
                    """Tensor type: {}, but torch or numpy array is 
                    expected!""".format(type(array))
                )
            plot_info = dict(env=env, win=win)
            self._log(level, "Histogram logged!", None,
                      extra=dict(
                          plot_info=plot_info,
                          data=array,
                          recordtype="histogram")
                      )

    def hyperparameters(self, params, env="main", win=None, **kwargs):
        """
        Log hyperparameters for better visualization.

        Arguments
            params: Dictionary of hyper-parameters and corresponding values.
            env: Name of the environment for Visdom (default "main")
            win: Name of the window (default None)
        """
        level = logging.INFO
        if self.isEnabledFor(level):
            plot_info = dict(env=env, win=win)
            self._log(level, "Hyperparameters logged!", None,
                      extra=dict(
                          plot_info=plot_info,
                          data=params,
                          recordtype="hyperparameters")
                      )

    def video(self, videofile, env="main", win=None, **kwargs):
        """
        Log a video by the given filename.

        Arguments
            videofile: Name of the video file.
            env: Name of the environment for Visdom (default "main")
            win: Name of the window (default None)
        """
        level = logging.INFO
        print(videofile)
        if self.isEnabledFor(level):
            plot_info = dict(env=env, win=win)
            self._log(level, "Video is logged", None,
                      extra=dict(
                          plot_info=plot_info,
                          data=videofile,
                          recordtype="video")
                      )


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