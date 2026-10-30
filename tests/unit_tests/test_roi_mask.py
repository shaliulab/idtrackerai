import numpy as np

from idtrackerai.animals_detection.segmentation_utils import (
    draw_roi_mask,
    get_roi_mask,
)
from idtrackerai.utils.conf_convert import parse_rois

SHAPE = (100, 100)
SQUARE = "[10, 10], [90, 10], [90, 90], [10, 90],"
HOLE = "[40, 40], [60, 40], [60, 60], [40, 60],"


def test_negative_roi_is_removed_from_positive_roi():
    mask = draw_roi_mask([[SQUARE], ["- " + HOLE]], SHAPE)
    assert mask[20, 20] == 255
    assert mask[50, 50] == 0
    assert mask[5, 5] == 0


def test_rois_without_sign_are_positive():
    assert (
        draw_roi_mask([[SQUARE]], SHAPE) == draw_roi_mask([["+ " + SQUARE]], SHAPE)
    ).all()


def test_only_negative_rois_start_from_full_frame():
    mask = draw_roi_mask([["- " + HOLE]], SHAPE)
    assert mask[5, 5] == 255
    assert mask[50, 50] == 0


def test_get_roi_mask_with_negative_roi():
    config = {
        "_applyroi": {"value": "True"},
        "_roi": {"value": [[SQUARE], ["- " + HOLE]]},
    }
    mask = get_roi_mask(config, SHAPE)
    assert mask[20, 20] == 255
    assert mask[50, 50] == 0


def test_conf_convert_negative_polygon():
    rows = parse_rois(
        [
            "+ Ellipse {'center': [50, 50], 'axes': [45, 45], 'angle': 0}",
            "- Polygon [[40.2, 40.0], [60.0, 40.0], [60.0, 60.0], [40.0, 60.0]]",
        ]
    )
    assert not rows[0].startswith("-")
    assert rows[1] == "- [40, 40], [60, 40], [60, 60], [40, 60], "
    mask = draw_roi_mask([[row] for row in rows], SHAPE)
    assert mask[30, 30] == 255
    assert mask[50, 50] == 0
