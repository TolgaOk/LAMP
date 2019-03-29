
from multiprocessing.sharedctypes import Value
from multiprocessing import Manager, Process
from collections import namedtuple
from collections import defaultdict


class SharedDefaultDict:
    def __init__(self, manager, default_factory=None):
        self.shared_dict = manager.dict()
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return self.shared_dict[key]
        except KeyError as e:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        value = self.default_factory()
        self.shared_dict[key] = value
        return self.shared_dict[key]

    def __setitem__(self, key, value):
        self.shared_dict[key] = value

    def __repr__(self):
        return self.shared_dict.__repr__()

    def __str__(self):
        return self.shared_dict.__str__()


class KeyDefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            self[key] = self.default_factory(key)
            return self[key]


class SharedKeyDefaultDict(SharedDefaultDict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = self.default_factory(key)
        return self[key]
