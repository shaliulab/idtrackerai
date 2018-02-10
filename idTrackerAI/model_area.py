from __future__ import absolute_import, division, print_function
from constants import STD_TOLERANCE

class ModelArea(object):
    """Model of the area used to perform a first discrimination between blobs
    representing single individual and multiple touching animals (crossings)

    Attributes
    ----------

    median : float
        median of the area of the blobs segmented from portions of the video in
        which all the animals are visible (not touching)
    mean : float
        mean of the area of the blobs segmented from portions of the video in
        which all the animals are visible (not touching)
    std : float
        standard deviation of the area of the blobs segmented from portions of
        the video in which all the animals are visible (not touching)
    std_tolerance : int
        tolerance factor
    """
    def __init__(self, mean, median, std):
        self.median = median
        self.mean = mean
        self.std = std
        self.std_tolerance = STD_TOLERANCE

    def __call__(self, area, std_tolerance = None):
        if std_tolerance is not None:
            self.std_tolerance = std_tolerance
        return (area - self.median) < self.std_tolerance * self.std
