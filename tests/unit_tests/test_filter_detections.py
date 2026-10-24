import cv2
from idtrackerai.animals_detection.yolov7.yolov7 import (
    load_detections_from_one_file,
    filter_detections_in_roi
)
allowed_classes={0: "fly", 1: "blurry"}

label_file="./test_filter_detections/2068277_45-43277.txt"
detections=load_detections_from_one_file(label_file, class_names=allowed_classes)
mask = cv2.imread("./test_filter_detections/mask.png")

assert len(detections)==7
print(detections)
detections=filter_detections_in_roi(detections, mask)
print(detections)
assert len(detections)==6