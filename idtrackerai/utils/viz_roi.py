import argparse
import json
import numpy as np
import cv2


def get_parser():
    ap=argparse.ArgumentParser()
    ap.add_argument("-i", dest="json_file", required=True)
    ap.add_argument("--image", dest="image", required=False, default=None)
    return ap

def load_roi(path):
    with open(path, "r") as handle:
        conf=json.load(handle)

    roi=np.array(eval("[" + conf["_roi"]["value"][0][0] + "]"))
    return roi

def main():
    ap=get_parser()
    args=ap.parse_args()
    
    roi=load_roi(args.json_file)
    roi[roi<0]=0
    if len(roi.shape)==2:
        roi=roi.reshape((roi.shape[0], 1, roi.shape[1]))


    if args.image is None:
        shape=tuple(roi.max(axis=0).flatten().tolist())
        mask=np.zeros(shape[::-1], dtype=np.uint8)
        mask=cv2.drawContours(mask, [roi], -1, 255, -1)
        cv2.imshow("ROI", mask)
        cv2.waitKey(0)
    else:
        image=cv2.imread(args.image)[:,:,0]
        shape=image.shape

        roi[:,0, 0][roi[:,0, 0] > shape[0]]=shape[0]-1
        roi[:,0, 1][roi[:,0, 1] > shape[1]]=shape[1]-1
        mask=np.zeros_like(image)
        mask=cv2.drawContours(mask, [roi], -1, 255, -1)
        print(image.shape, mask.shape)
    
        image=cv2.addWeighted(image, 0.5, mask, 0.5, 0.0)
        cv2.imshow("ROI", image)
        cv2.waitKey(0)


    


if __name__ == "__main__":
    main()
