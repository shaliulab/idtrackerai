from idtrackerai.animals_detection.segmentation_utils import _get_pixels

import numpy as np


def test_get_pixels():
    min_x, max_x = 1, 5
    min_y, max_y = 1, 3
    num_pixels = (max_x - min_x + 1) * (max_y - min_y + 1)
    expected_pixels = np.asarray(
        [
            [i, j]
            for i in range(min_y, max_y + 1)
            for j in range(min_x, max_x + 1)
        ]
    )
    width = 10
    height = 10
    cnt = np.asarray(
        [[[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]]]
    )
    pixels = _get_pixels(cnt, width, height)
    print(pixels, expected_pixels)
    assert isinstance(pixels, np.ndarray)
    assert pixels.dtype == np.int64
    assert pixels.shape[0] == num_pixels
    assert pixels.shape[1] == 2
    np.testing.assert_equal(pixels, expected_pixels)
