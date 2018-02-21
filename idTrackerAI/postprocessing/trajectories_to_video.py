from __future__ import absolute_import, print_function, division
import numpy as np
import cv2
import sys
sys.path.append('../')
sys.path.append('../utils')
from py_utils import get_spaced_colors_util


def init(video_path, trajectories_dict_path):
    video_object = np.load(video_path).item()
    video_object.update_paths(video_path)
    trajectories  = np.load(trajectories_dict_path).item()['trajectories']
    colors = get_spaced_colors_util(video_object.number_of_animals)
    path_to_save_video = video_object._session_folder +'/tracked.avi'
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    video_writer = cv2.VideoWriter(path_to_save_video, fourcc, video_object.frames_per_second,
                                    (video_object.width, video_object.height))
    print("The video will be saved at ", path_to_save_video)
    return video_object, trajectories, video_writer, colors

def generate_video(video_object,
                trajectories,
                video_writer,
                colors,
                func = None,
                centroid_trace_length = 10):
    if video_object.paths_to_video_segments is None:
        cap = cv2.VideoCapture(video_object.video_path)
    else:
        cap = None


    for frame_number in range(len(trajectories)):
        frame = apply_func_on_frame(video_object,
                            frame_number,
                            video_writer,
                            colors,
                            cap = cap,
                            func = writeIds,
                            centroid_trace_length  = centroid_trace_length)
        video_writer.write(frame)

def apply_func_on_frame(video_object,
                        frame_number,
                        video_writer,
                        colors,
                        cap = None,
                        func = None,
                        centroid_trace_length = 10):
    segment_number = video_object.in_which_episode(frame_number)
    current_segment = segment_number
    if cap is None:
        cap = cv2.VideoCapture(video_object.paths_to_video_segments[segment_number])
        start = video_object._episodes_start_end[segment_number][0]
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,frame_number - start)
    else:
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    print(ret)
    if ret:
        if video_object.resolution_reduction != 1:
            frame = cv2.resize(frame, None,
                                fx = video_object.resolution_reduction,
                                fy = video_object.resolution_reduction)
        frame = func(frame, frame_number, trajectories, centroid_trace_length,
                    colors)
        return frame


def writeIds(frame, frame_number, trajectories, centroid_trace_length, colors):
    ordered_centroid = trajectories[frame_number]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for cur_id, centroid in enumerate(ordered_centroid):
        if sum(np.isnan(centroid)) == 0:
            if frame_number > centroid_trace_length:
                centroids_trace = trajectories[frame_number - centroid_trace_length : frame_number, cur_id]
            else:
                centroids_trace = trajectories[: frame_number, cur_id]
            cur_id_str = str(cur_id + 1)
            int_centroid = np.asarray(centroid).astype('int')
            cv2.circle(frame, tuple(int_centroid), 2, colors[cur_id], -1)
            cv2.putText(frame, cur_id_str,tuple(int_centroid), font, 1, colors[cur_id], 3)
            for centroid_trace in centroids_trace:
                if sum(np.isnan(centroid_trace)) == 0:
                    int_centroid = np.asarray(centroid_trace).astype('int')
                    cv2.circle(frame, tuple(int_centroid), 2, colors[cur_id], -1)
    return frame


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video_object_path", type = str, help = "Path to the video object created during the tracking session")
    parser.add_argument("-t", "--trajectories_path", type = str, help = "Path to the trajectory file")
    parser.add_argument("-s", "--number_of_ghost_points", type = int, default = 20,
                        help = "Number of points used to draw the individual trajectories' traces")
    args = parser.parse_args()
    video_object_path = args.video_object_path
    trajectories_dict_path = args.trajectories_path
    number_of_points_in_past_trace = args.number_of_ghost_points
    video_object, trajectories, video_writer, colors = init(video_object_path,
                                                    trajectories_dict_path)
    generate_video(video_object,
                    trajectories,
                    video_writer,
                    colors,
                    func = writeIds,
                    centroid_trace_length = number_of_points_in_past_trace)
