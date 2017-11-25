import math

class Sample(object):
    def __init__(self, feature = [], label = 0):
        self._features = feature
        self._label = label
        self._predict = 0


class LogisticRegression(object):
    def __init__(self):
        self.__reset()

    def __reset(self):
        self._w = []
        self._b = 0.0
        self._feature_size = 0

    def prepare(self, feature_size):
        assert int(feature_size) > 0
        self._feature_size = int(feature_size)
        self._w = [0.0] * self._feature_size

    def fit(self, sample_list):
        assert isinstance(sample_list, list)
