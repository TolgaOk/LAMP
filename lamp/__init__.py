import matplotlib
matplotlib.use("TkAgg")

from .logger import DataLogger
from .matplothandlers import MatplotScalarHandler
from .matplothandlers import MatplotImageHandler
from .vizdomhandlers import VisdomScalarHandler
from .vizdomhandlers import VisdomImageHandler
from .vizdomhandlers import VisdomHistHandler
from .vizdomhandlers import VisdomParameterHandler
from .vizdomhandlers import VisdomVideoHandler

__all__ = ["DataLogger", "MatplotScalarHandler", "MatplotImageHandler",
        "VisdomScalarHandler", "VisdomImageHandler", "VisdomHistHandler",
        "VisdomParameterHandler"]