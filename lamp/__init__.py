""" 
"""
import logging

import matplotlib
matplotlib.use("TkAgg")

from lamp.vizdomhandlers import VisdomVideoHandler
from lamp.vizdomhandlers import VisdomParameterHandler
from lamp.vizdomhandlers import VisdomHistHandler
from lamp.vizdomhandlers import VisdomImageHandler
from lamp.vizdomhandlers import VisdomScalarHandler
from lamp.matplothandlers import MatplotImageHandler
from lamp.matplothandlers import MatplotScalarHandler
from lamp.logger import DataLogger

__all__ = ["DataLogger", "MatplotScalarHandler", "MatplotImageHandler",
           "VisdomScalarHandler", "VisdomImageHandler", "VisdomHistHandler",
           "VisdomParameterHandler"]
logging.root.manager.setLoggerClass(DataLogger)
