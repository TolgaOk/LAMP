from lamp import MatplotScalarHandler
from lamp import MatplotImageHandler
from lamp import DataLogger
from lamp import VisdomScalarHandler
from lamp import VisdomImageHandler
from lamp import VisdomHistHandler
from lamp import VisdomParameterHandler

import logging
import numpy as np
import torch

def matplothandlers():
    logging.setLoggerClass(DataLogger)
    my_logger = logging.getLogger("main")

    handlerscalar = MatplotScalarHandler()
    handlerimage = MatplotImageHandler()

    my_logger.addHandler(handlerscalar)
    my_logger.addHandler(handlerimage)

    my_logger.setLevel(logging.DEBUG)

    my_logger.image(np.random.uniform(size=(32, 32)), cmap="gray")
    for i in range(50):
        my_logger.scalar(i**2)
    
def vizdomhandlers():

    logging.setLoggerClass(DataLogger)
    my_logger = logging.getLogger(__name__)

    vizscalarhandler = VisdomScalarHandler(logging.INFO,
                                        overwrite_window=True,
                                        lock=None)
    vizhisthandler = VisdomHistHandler(logging.DEBUG,
                                        overwrite_window=True)

    my_logger.addHandler(vizscalarhandler)
    my_logger.addHandler(vizhisthandler)
    my_logger.setLevel(logging.DEBUG)


    for i in range(10):
        my_logger.scalar(i**2, win="polynomial", trace="x^2")
        my_logger.scalar(i**2-5*i + 2, win="polynomial", trace="x^2 - 5x + 2")
    
    gauss_vector = torch.randn(500)
    my_logger.histogram(gauss_vector, win="Normal distribution")
    
if __name__ == "__main__":
    matplothandlers()
    matplothandlers()