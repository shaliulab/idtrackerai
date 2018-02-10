from __future__ import absolute_import, print_function, division
import numpy as np
import cv2
import sys
sys.path.append('../')
sys.path.append('../utils')
from py_utils import get_spaced_colors_util


def init(video_path, trajectories_dict_path):
    video_object = np.load(video_path).item()
    trajectories  = np.load(trajectories_dict_path).item()['trajectories']
    colors = get_spaced_colors_util(video_object.number_of_animals)
    path_to_save_video = video_object._session_folder +'/tracked.avi'
    video_writer = cv2.VideoWriter(path_to_save_video, fourcc, 32.0,
                                    (video_object.width, video_object.height))
    print("The video will be saved at ", path_to_save_video)
    return video_object, trajectories, video_writer

def generate_video(video_object,
                trajectories,
                video_writer,
                func = None,
                centroid_trace_length = 10):

    for frame_number in range(len(trajectories)):
        frame = apply_func_on_frame(video_object,
                            frame_number,
                            video_writer,
                            func = writeIds,
                            centroid_trace_length = 10)
        video_writer.write(frame)

def apply_func_on_frame(video_object,
                        frame_number,
                        video_writer,
                        func = None,
                        centroid_trace_length = 10):
    segment_number = video_object.in_which_episode(frame_number)
    current_segment = segment_number
    if video_object.paths_to_video_segments:
        cap = cv2.VideoCapture(video_object.paths_to_video_segments[segment_number])
        start = video_object._episodes_start_end[segment_number][0]
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,frame_number - start)
    else:
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        if video_object.resolution_reduction != 1:
            frame = cv2.resize(frame, None,
                                fx = video_object.resolution_reduction,
                                fy = video_object.resolution_reduction)
        frame = func(frame, frame_number, trajectories, centroid_trace_length)
        return frame


def writeIds(frame, frame_number, trajectories, centroid_trace_length):
    ordered_centroid = trajectories[frame_number]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for cur_id, centroid in enumerate(ordered_centroid):
        if sum(np.isnan(centroid)) == 0:
            if frame_number > centroid_trace_length:
                centroids_trace = trajectories[frame_number - centroid_trace_length : frame_number]
            else:
                centroids_trace = trajectories[: frame_number]
            cur_id_str = str(cur_id)
            int_centroid = np.asarray(centroid).astype('int')
            cv2.circle(frame, tuple(int_centroid), 2, colors[cur_id], -1)
            cv2.putText(frame, cur_id_str,tuple(int_centroid), font, 1, colors[cur_id], 3)
            for centroid_trace in centroids_trace:
                int_centroid = np.asarray(centroid_trace).astype('int')
                cv2.circle(frame, tuple(centroid_trace), 2, colors[cur_id], -1)
    return frame


if __name__ == "__main__":
    video_path = '/Users/mattiagiuseppebergomi/Desktop/test/video_object.npy'
    trajectories_dict_path = '/Users/mattiagiuseppebergomi/Desktop/test/trajectories.npy'
    video_object, trajectories, video_writer = init(video_path,
                                                    trajectories_dict_path)
    generate_video(video_object,
                    trajectories,
                    video_writer,
                    func = writeIds)
