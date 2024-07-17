import argparse
import json
from pathlib import Path
import tomli

import numpy as np
import cv2

def get_parser():
    ap=argparse.ArgumentParser()
    ap.add_argument("-i", dest="toml_file", required=True)
    ap.add_argument("-o", dest="json_file", required=True)
    return ap

def main():
    
    ap=get_parser()
    args=ap.parse_args()
    conf=tomli.loads(Path(args.toml_file).read_text(encoding="utf-8"))

    roi_str=parse_roi(conf["roi_list"][0])
    conf_json=generate_v4_conf(conf["intensity_ths"][0], conf["intensity_ths"][1], conf["area_ths"][0], conf["area_ths"][1], conf["number_of_animals"], roi_str=roi_str)
    with open(args.json_file, "w") as handle:
        json.dump(conf_json, handle)
    

def parse_roi(roi):

    ellipse_roi=eval(roi.lstrip("+ Ellipse "))
    center=tuple(ellipse_roi["center"])
    axes=tuple(ellipse_roi["axes"])
    angle=ellipse_roi["angle"]

    # Create a blank image, size should be large enough to contain the ellipse
    # Here, I am assuming a size based on the ellipse dimensions and center; you might adjust this as needed
    image_size = (int(center[0] + axes[0]*2), int(center[1] + axes[1]*2), 3)
    image = np.zeros(image_size, dtype=np.uint8)

    # Color and thickness for the ellipse (using white color and thickness -1 to fill the ellipse)
    color = (255, 255, 255) # White color
    thickness = -1 # Fill

    # Draw the ellipse on the blank image
    cv2.ellipse(image, center, axes, angle, 0, 360, color, thickness)

    # Convert to grayscale for findContours
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find contours
    _, contours, _= cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour=contours[0]
    n_points=contour.shape[0]
    contour=contour[::n_points//50, ...]
    contour[contour<0]=0
    contour_str=", ".join([str(e) for e in contour[:, 0, :].tolist()]) + ", "
    return contour_str


def generate_v4_conf(min_intensity, max_intensity, min_area, max_area, number_of_animals, roi_str):
    "[1180, 652], [1180, 652],"

    conf={
        "open-multiple-files": False,
        "_video": {
            "value": "../metadata.yaml"
        },
        "_range": {
            "value": [
                0,
                99999999
            ],
            "max": 99999999,
            "min": 0,
            "scale": 1.0,
            "convert-int": False
        },
        "_rangelst": {
            "value": ""
        },
        "_multiple_range": {
            "value": "False"
        },
        "_intensity": {
            "value": [
                min_intensity,
                max_intensity
            ],
            "max": 255,
            "min": 0,
            "scale": 1.0,
            "convert-int": False
        },
        "_area": {
            "value": [
                min_area,
                max_area
            ],
            "max": 60000,
            "min": 0,
            "scale": 1.0,
            "convert-int": False
        },
        "_number_of_animals": {
            "value": number_of_animals
        },
        "_resreduct": {
            "value": 1.0
        },
        "_chcksegm": {
            "value": "False"
        },
        "_roi": {
            "value": [
                [
                    roi_str
                ]
            ]
        },
        "_no_ids": {
            "value": "False"
        },
        "_applyroi": {
            "value": "True"
        },
        "_bgsub": {
            "value": "False"
        },
        "_add_setup_info": {
            "value": "False"
        },
        "_points_list": {}
    }

    return conf

if __name__ == "__main__":
    main()

