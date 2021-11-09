# This file is part of idtracker.ai a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
# Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
# Champalimaud Foundation.
#
# idtracker.ai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (idtrackerai@gmail.com) or
# use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.
#
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H.,
# de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of
# unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P:
# gonzalo.polavieja@neuro.fchampalimaud.org)

import glob
import logging
import multiprocessing
import os
import re

import numpy as np
from matplotlib import cm

logger = logging.getLogger("__main__.py_utils")

### MKL
def set_mkl_to_single_thread():
    logger.info("Setting MKL library to use single thread")
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_DYNAMIC"] = "FALSE"


def set_mkl_to_multi_thread():
    logger.info("Setting MKL library to use multiple threads")
    os.environ["MKL_NUM_THREADS"] = str(multiprocessing.cpu_count())
    os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
    os.environ["MKL_DYNAMIC"] = "TRUE"


### Object utils ###
def append_values_to_lists(values, list_of_lists):
    list_of_lists_updated = []

    for list_, value in zip(list_of_lists, values):
        list_.append(value)
        list_of_lists_updated.append(list_)

    return list_of_lists_updated


def set_attributes_of_object_to_value(
    object_to_modify, attributes_list, value=None
):
    [
        setattr(object_to_modify, attribute, value)
        for attribute in attributes_list
        if hasattr(object_to_modify, attribute)
    ]


def delete_attributes_from_object(object_to_modify, list_of_attributes):
    [
        delattr(object_to_modify, attribute)
        for attribute in list_of_attributes
        if hasattr(object_to_modify, attribute)
    ]


### Dict utils ###
def flatten(list_):
    """flatten a list of lists"""
    try:
        ans = [inner for outer in list_ for inner in outer]
    except TypeError:
        ans = [y for x in list_ for y in (x if isinstance(x, tuple) else (x,))]
    return ans


def get_spaced_colors_util(n, norm=False, black=True, cmap="jet"):
    RGB_tuples = cm.get_cmap(cmap)
    if norm:
        colors = [RGB_tuples(i / n) for i in range(n)]
    else:
        RGB_array = np.asarray([RGB_tuples(i / n) for i in range(n)])
        BRG_array = np.zeros(RGB_array.shape)
        BRG_array[:, 0] = RGB_array[:, 2]
        BRG_array[:, 1] = RGB_array[:, 1]
        BRG_array[:, 2] = RGB_array[:, 0]
        colors = [tuple(BRG_array[i, :] * 256) for i in range(n)]
    if black:
        black = (0.0, 0.0, 0.0)
        colors.insert(0, black)
    return colors


def interpolate_nans(t):
    """Interpolates nans linearly in a trajectory

    :param t: trajectory
    :returns: interpolated trajectory
    """
    shape_t = t.shape
    reshaped_t = t.reshape((shape_t[0], -1))
    for timeseries in range(reshaped_t.shape[-1]):
        y = reshaped_t[:, timeseries]
        nans, x = _nan_helper(y)
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])

    # Ugly slow hack, as reshape seems not to return a view always
    back_t = reshaped_t.reshape(shape_t)
    t[...] = back_t


def _nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.
    https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]
