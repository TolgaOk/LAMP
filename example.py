from lamp import MatplotScalarHandler
from lamp import MatplotImageHandler
from lamp import DataLogger
from lamp import VisdomScalarHandler
from lamp import VisdomImageHandler
from lamp import VisdomHistHandler
from lamp import VisdomParameterHandler

import logging
import numpy as np

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

    vizscalarhandler = VisdomScalarHandler(logging.DEBUG, True, lock=None)
    
    my_logger.addHandler(vizscalarhandler)
    my_logger.setLevel(logging.DEBUG)

    for i in range(10):
        my_logger.scalar(i**2, win="loss", trace="trace-1")
        my_logger.scalar(i**2-5*i + 2, win="loss", trace="trace-2")

vizdomhandlers()