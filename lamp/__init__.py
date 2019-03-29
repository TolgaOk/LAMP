from .vizdomhandlers import VisdomVideoHandler
from .vizdomhandlers import VisdomParameterHandler
from .vizdomhandlers import VisdomHistHandler
from .vizdomhandlers import VisdomImageHandler
from .vizdomhandlers import VisdomScalarHandler
from .matplothandlers import MatplotImageHandler
from .matplothandlers import MatplotScalarHandler
from .logger import DataLogger

# import matplotlib
# matplotlib.use("TkAgg")


__all__ = ["DataLogger", "MatplotScalarHandler", "MatplotImageHandler",
           "VisdomScalarHandler", "VisdomImageHandler", "VisdomHistHandler",
           "VisdomParameterHandler"]
