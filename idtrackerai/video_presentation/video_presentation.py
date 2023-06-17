import re
import math
import argparse
import os.path
import yaml
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import cv2
import joblib
# FLYHOSTEL_VIDEOS=os.environ["FLYHOSTEL_VIDEOS"]
FLYHOSTEL_VIDEOS="/Users/FlySleepLab Dropbox/Data/flyhostel_data/videos/"

from imgstore.interface import VideoCapture

LIVE_WRITING=False
RESOLUTION=(1000, 1000)


# data
# chunk, frame_number, experiment, roi, text, center, box, color, frame_index, identity, video
fontFace=cv2.FONT_HERSHEY_SIMPLEX
fontScale=2
right_text=(650, 100)


COLORS={
    "white": (255, 255, 255),
    "green": (0, 255, 0),
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
}

def apply_roi(frame, roi):

    x, y, w, h = roi
    return frame[y:(y+h), x:(x+w)]


def roi_from_data(data, suffix=None):

    columns = ["x", "y", "w", "h"]
    if suffix is not None:
        columns = [e + f"_{suffix}" for e in columns]
    
    roi = data[columns]
    if all(pd.isna(roi)):
        roi=None
    else:
        roi=tuple(roi.values.tolist())

    return roi


def draw_box(frame, box, color, text, thickness):
    x, y, w, h = box
    frame=cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness=thickness)
    cv2.copyMakeBorder(frame, 50, 50, 50, 50, cv2.BORDER_CONSTANT, frame, 255)

    if not np.isnan(text):
        (width, height), baseline = cv2.getTextSize(text, fontFace, fontScale, thickness)
        width=int(width*1.1)
        height=int(height*1.1)
        tl_corner = (int(box[2]*0.9), int(box[3]*0.9))
        tl_corner = (tl_corner[0], tl_corner[1]-height)
        br_corner = (tl_corner[0]+width, box[1])
        frame = cv2.rectangle(frame, tl_corner, br_corner, COLORS[color], -1)
        frame = cv2.putText(frame, text, tl_corner, fontFace, fontScale, COLORS["white"], 1)

    return frame

def draw_text(frame, color, text, pos, thickness=6):

    (width, height), baseline = cv2.getTextSize(text, fontFace, fontScale, thickness)
    bl_corner = pos
    tl_corner = (pos[0], pos[1]-height)
    br_corner = (pos[0]+width, pos[1])
    # print(tl_corner)
    # print(br_corner)

    # frame = cv2.rectangle(frame, tl_corner, br_corner, COLORS[color], -1)
    frame=cv2.putText(frame, text, pos, fontFace, fontScale, COLORS["white"], thickness)
    return frame


def annotate_frame(frame, data):

    for i, row in data.iterrows():
        box =roi_from_data(data, "box")
        if box is not None:
            frame = draw_box(frame, box, row["color"], row["text"], thickness=2)
        
    return frame

def annotate_frame_resized(frame, data):
    for i, row in data.iterrows():
        frame = draw_text(frame, row["color"], row["text"], (50, 100))
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

    roi=roi_from_data(data)
    frameSize=RESOLUTION

    filename=data["video"]

    vw = cv2.VideoWriter(
        filename,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps=fps,
        frameSize=frameSize,
        isColor=True,
    )
    return vw


def build_store_path_from_experiment(experiment):
    flyhostel_id = int(re.search("FlyHostel([0-9]).*", experiment).group(1))
    number_of_animals = int(re.search("FlyHostel[0-9]_([0-9][0-9]?)X.*", experiment).group(1))
    date_time = re.search(
        "FlyHostel[0-9]_[0-9][0-9]?X_([0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]-[0-9][0-9]-[0-9][0-9])",
        experiment
    ).group(1)


    folder_structure = os.path.join(f"FlyHostel{flyhostel_id}", f"{number_of_animals}X", date_time)
    store_path = os.path.join(FLYHOSTEL_VIDEOS, folder_structure, "metadata.yaml")
    return store_path


def format_seconds(seconds):
    minutes = int(seconds // 60)
    seconds = round(seconds % 60)
    mm_ss=f"{str(minutes).zfill(2)}:{str(seconds).zfill(2)}"
    return mm_ss

def make_video(data, fps=150, before=0, after=0):
    
    experiment=data["experiment"].iloc[0]
    
    store_path = build_store_path_from_experiment(experiment)
    with open(store_path, "r") as filehandle:
        metadata = yaml.load(filehandle, yaml.SafeLoader)
        input_fps = metadata["__store"]["framerate"]

        before*=input_fps
        after*=input_fps


    n_reps = int(math.ceil(input_fps/fps))
    cap = VideoCapture(store_path, data.iloc[0]["chunk"].item())
    
    video_writer=init_video_writer(data.iloc[0], fps=input_fps*5)
    last_frame_number = -10

    frame_numbers=sorted(list(set(data["frame_number"].values.tolist())))
    assert all(np.diff(frame_numbers) == 1), "Videos with jumps are not supported"
    assert len(np.unique(data["experiment"])) == 1
    
    frame_number=frame_numbers[0]-before
    cap.set(1, frame_number)
    t0=cap.frame_time/1000
    (width, height), baseline = cv2.getTextSize("00:00", fontFace, fontScale, 6)
    timer_pos = (int(500 - width / 2), height+50)




    pb=tqdm(total=before, position=0)
    ret=True
    while frame_number < frame_numbers[0] and ret:
        ret, frame = cap.read()
        seconds=cap.frame_time/1000-t0
        mm_ss = format_seconds(seconds)

        roi=roi_from_data(data.loc[data["frame_number"] == frame_numbers[0]].iloc[0])
        if roi is not None:
            frame = apply_roi(frame, roi)
        frame = cv2.resize(frame, RESOLUTION, cv2.INTER_AREA)
        frame=cv2.putText(frame, mm_ss, timer_pos, fontFace, fontScale, COLORS["white"], 6)
        video_writer.write(frame)
        pb.update(1)
        frame_number+=1

    pb=tqdm(total=len(frame_numbers), position=1)
    for frame_number in frame_numbers:
        if (last_frame_number+1) == frame_number:
            pass
        else:
            cap.set(1, frame_number)

        last_frame_number=frame_number
        ret, frame = cap.read()
        seconds=cap.frame_time/1000-t0
        mm_ss = format_seconds(seconds)

        frame_orig = frame.copy()
        frame = annotate_frame(frame, data.loc[data["frame_number"] == frame_number])
        roi=roi_from_data(data.loc[data["frame_number"] == frame_number].iloc[0])
        if roi is not None:
            frame = apply_roi(frame, roi)

        frame = cv2.resize(frame, RESOLUTION, cv2.INTER_AREA)
        frame=annotate_frame_resized(frame, data.loc[data["frame_number"] == frame_number])
        
        if LIVE_WRITING:
            cv2.imshow("frame", frame)
            cv2.waitKey(1)

        frame=cv2.putText(frame, "3X slower", right_text, fontFace, fontScale, COLORS["white"], 6)
        frame=cv2.putText(frame, mm_ss, timer_pos, fontFace, fontScale, COLORS["white"], 6)

        for _ in range(n_reps):
            video_writer.write(frame)
        pb.update(1)


    if after:
        pb=tqdm(total=after, position=2)
        while frame_number < (frame_numbers[-1]+after) and ret:
            ret, frame = cap.read()
            seconds=cap.frame_time/1000-t0
            mm_ss = format_seconds(seconds)

            roi=roi_from_data(data.loc[data["frame_number"] == frame_numbers[0]].iloc[0])
            if roi is not None:
                frame = apply_roi(frame, roi)
            frame = cv2.resize(frame, RESOLUTION, cv2.INTER_AREA)
            
            frame=cv2.putText(frame, mm_ss, timer_pos, fontFace, fontScale, COLORS["white"], 6)
            video_writer.write(frame)
            pb.update(1)
            frame_number+=1



    cap.release()
    video_writer.release()
    return frame_orig



def read_user_data(annotation):
    data=pd.read_csv(annotation)
    data.sort_values(["experiment", "chunk", "frame_number"])
    return data