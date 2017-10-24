from __future__ import absolute_import, division, print_function

STD_TOLERANCE = 4 ### NOTE set to 1 because we changed the model area to work with the median.
class ModelArea(object):
    def __init__(self, mean, median, std):
        self.median = median
        self.mean = mean
        self.std = std
        self.std_tolerance = STD_TOLERANCE

    def __call__(self, area, std_tolerance = None):
        if std_tolerance is not None:
            self.std_tolerance = std_tolerance
        return (area - self.median) < self.std_tolerance * self.std
