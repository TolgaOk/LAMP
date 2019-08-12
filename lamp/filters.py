import logging
from lamp.logger import DataLogger


class DataLogFilter(logging.Filter):
    def __init__(self, filtertype, name=""):
        super().__init__(name=name)
        self.filtertype = filtertype

    def filter(self, record):
        try:
            record.plot_info
            return record.recordtype == self.filtertype
        except AttributeError:
            return False


class NonDataLogFilter(logging.Filter):
    def __init__(self, name=""):
        super().__init__(name=name)

    def filter(self, record):
        try:
            record.plot_info
            return False
        except AttributeError:
            return True


class CheckConnection(logging.Filter):
    def __init__(self, connection, name=""):
        super().__init__(name)
        self.connection = connection

    def filter(self):
        pass
