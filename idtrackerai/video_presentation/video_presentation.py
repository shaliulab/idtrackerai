import re
import argparse
import os.path

import pandas as pd
import numpy as np
import cv2
import joblib

from imgstore.interface import VideoCapture

LIVE_WRITING=False

def apply_roi(frame, roi):

    x, y, w, h = roi
    return frame[y:(y+h), x:(x+w)]

# data
# chunk, frame_number, experiment, roi, text, center, box, color, frame_index, identity, video
fontFace=cv2.FONT_HERSHEY_SIMPLEX
fontScale=2

def draw_box(frame, box, color, text, thickness):
    x, y, w, h = box
    frame=cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness=thickness)
    cv2.copyMakeBorder(frame, 50, 50, 50, 50, cv2.BORDER_CONSTANT, frame, 255)

    if not np.isnan(text):
    

        offset_y = int(frame.shape[0] * 0.01)
        (width, height), baseline = cv2.getTextSize(text, fontFace, fontScale, thickness)
        tl_corner = box[:2]
        tl_corner = (tl_corner[0], tl_corner[1]-height)
        br_corner = (tl_corner[0]+width, box[1])
        frame = cv2.rectangle(frame, tl_corner, br_corner, color, -1)
        frame = cv2.putText(frame, text, tl_corner, fontFace, fontScale, "white", 1)

    return frame

def draw_text(frame, color, text, roi=None):

    if roi is not None:
        frame=apply_roi(frame, roi)

    frame=cv2.putText(frame, text, (10, 10), fontFace, fontScale, color, 1)
    return frame


def annotate_frame(frame_number, frame, data):

    current_data = data.loc[data["frame_number"] == frame_number]
    for i, row in current_data.iterrows():
        if np.isnan(row["box"]):
            roi = row[["x", "y", "w", "h"]]
            if all(np.isnan(roi)):
                roi=None
            else:
                roi=roi.values.tolist()

            frame = draw_text(frame, row["color"], row["text"], roi)
        else:
            frame = draw_box(frame, row["box"], row["color"], row["text"], thickness=2)
        

        if not np.isnan(row["marker"]):
            raise NotImplementedError()
        
    return frame

def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", required=True)
    ap.add_argument("--chunk", default=50, type=int)
    ap.add_argument("--n-jobs", default=3, type=int)
    ap.add_argument("--annotation", type=str)
    return ap


def main():

    ap = get_parser()
    args = ap.parse_args()
    assert os.path.exists(args.annotation)
    data = read_user_data(args.annotation)

    experiments = list(set(data["experiment"].value.tolist()))

    joblib.Parallel(n_jobs=args.n_jobs)(
        joblib.delayed(
            make_video(data.loc[data["experiment"] == experiment])
        )
        for experiment in experiments
    )


def init_video_writer(data, fps=150):

    frameSize=data["roi"][2:]
    filename=data["video"]

    vw = cv2.VideoWriter(
        filename,
        cv2.VideoWriter_fourcc(*"MP4V"),
        frameSize=frameSize,
        fps=fps,
        isColor=True
    )
    return vw


def build_store_path_from_experiment(experiment):
    flyhostel_id = int(re.search("Flyhostel([0-9]).*", experiment).group(1))
    number_of_animals = int(re.search("Flyhostel[0-9]_([0-9][0-9]?)X.*", experiment).group(1))
    date_time = int(re.search(
        "Flyhostel[0-9]_([0-9][0-9]?)X_([0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]",
        experiment
    ).group(1))


    folder_structure = os.path.join(f"FlyHostel{flyhostel_id}", f"{number_of_animals}X", date_time)
    store_path = os.path.join(os.environ["FLYHOSTEL_VIDEOS"], folder_structure, "metadata.yaml")
    return store_path

def make_video(data):

    store_path = build_store_path_from_experiment(data["experiment"])
    cap = VideoCapture(store_path, data["chunk"])
    ret=True
    video_writer=init_video_writer(data)
    last_frame_number = -10

    frame_numbers=sorted(list(set(data["frame_number"].values.tolist())))
    assert all(np.diff(frame_numbers) == 1), "Videos with jumps are not supported"
    assert len(np.unique(data["experiment"])) == 1

    for frame_number in frame_numbers:
        if (last_frame_number+1) == frame_number:
            pass
        else:
            cap.set(1, frame_number)

        last_frame_number=frame_number
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1000, 1000), cv2.INTER_AREA)
        frame = annotate_frame(frame_number, frame, data)
        
        roi = data.loc[data["frame_number"] == frame_number, ["x", "y", "w", "h"]].iloc[0]
        if all(np.isnan(roi)):
            roi=None
        else:
            roi=roi.values.tolist()

        frame = apply_roi(frame, roi)
        if LIVE_WRITING:
            cv2.imshow("frame", frame)
            cv2.waitKey(1)
        video_writer.write(frame)

    cap.release()
    video_writer.release()


def read_user_data(annotation):
    data=pd.read_csv(annotation)
    data.sort_values(["experiment", "chunk", "frame_number"])
    return data