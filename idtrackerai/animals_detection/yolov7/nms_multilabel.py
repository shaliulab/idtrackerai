import itertools
import numpy as np
import cv2


def draw_detection_mask(detection, mask):
    contour = np.array(detection.bounding_box).reshape((2, 1, 2))
    contour[1,:] += contour[0,:]
    # print(detection, contour)

    assert mask.shape[0] == mask.shape[1]
    contour *= mask.shape[0]

    contour=np.int32(np.round(contour))

    mask = cv2.drawContours(mask, [contour], -1, 255, -1)

    return mask

def compute_overlap_fraction(detection1, detection2):

    mask1 = np.zeros((100, 100), np.uint8)
    mask2 = mask1.copy()

    mask1 = draw_detection_mask(detection1, mask1)
    mask2 = draw_detection_mask(detection2, mask2)

    ov12=(cv2.bitwise_and(mask1, mask2) == 255).sum() / (mask1==255).sum()
    ov21=(cv2.bitwise_and(mask1, mask2) == 255).sum() / (mask2==255).sum()

    return ov12, ov21


def resolve_multilabel(detections):

    for detection1, detection2 in itertools.combinations(detections, 2):
        ov12, ov21 = compute_overlap_fraction(detection1, detection2)
        if ov12 > 0.9 and ov21 > 0.9:
            for detection in [detection1, detection2]:
                if detection.class_id == 0:
                    detection.keep = False
                elif detection.class_id == 1:
                    detection.keep = True
                    detection.overwrites = True


    final_detections = []
    for detection in detections:
        if detection.keep:
            final_detections.append(detection)

    return final_detections
