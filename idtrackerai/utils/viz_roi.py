import argparse
import json
import numpy as np
import cv2
from idtrackerai.animals_detection.segmentation_utils import parse_rois, draw_roi_mask


def get_parser():
    ap=argparse.ArgumentParser()
    ap.add_argument("-i", dest="json_file", required=True)
    ap.add_argument("--image", dest="image", required=False, default=None)
    return ap

def load_rois(path):
    with open(path, "r") as handle:
        conf=json.load(handle)

    return conf["_roi"]["value"]

def main():
    ap=get_parser()
    args=ap.parse_args()

    rois=load_rois(args.json_file)

    if args.image is None:
        contours=[roi_contour for _, roi_contour in parse_rois(rois)]
        shape=tuple((np.concatenate(contours).max(axis=0).flatten()+1).tolist())
        mask=draw_roi_mask(rois, shape[::-1])
        cv2.imshow("ROI", mask)
        cv2.waitKey(0)
    else:
        image=cv2.imread(args.image)[:,:,0]
        mask=draw_roi_mask(rois, image.shape)
        print(image.shape, mask.shape)

        image=cv2.addWeighted(image, 0.5, mask, 0.5, 0.0)
        cv2.imshow("ROI", image)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
