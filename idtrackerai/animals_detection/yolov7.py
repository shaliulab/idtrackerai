import os.path
import json

import cv2
import numpy as np

from .segmentation import _create_blobs_objects
from .segmentation_utils import _get_blobs_information_per_frame

yolov7_repo = "/scratch/leuven/333/vsc33399/Projects/YOLOv7/yolov7"

def yolo_line_to_detection(line):
    """
    
    Argumennts:
    
    * line (str): Label as read from a yolo .txt file
    
    Returns
    
    * detection (dict): class (int), bounding_box (x, y, w, h) and confidence (0 to 1)
    """
    
    line=line.split(" ")
    
    # class id
    line[0] = int(line[0])
    
    # bounding_box and confidence optinally
    for i in range(1, len(line)):
        line[i] = float(line[i])
    
    bounding_box = (line[1]-line[3]/2, line[2]-line[4]/2, *line[3:5])
    
    detection = {
        "class": line[0],
        "bounding_box": bounding_box
    }
    
    if len(line) == 6:
        detection["confidence"] =line[5]
    
    return detection

def detection2mask(detection, frame):
    bbox_norm = detection["bounding_box"]
    bbox = [
        bbox_norm[0] * frame.shape[1],
        bbox_norm[1] * frame.shape[0],
        bbox_norm[2] * frame.shape[1],
        bbox_norm[3] * frame.shape[0]
    ]

    bbox = tuple([int(round(coord)) for coord in bbox])
    

    detection_mask = np.zeros_like(frame)
    detection_mask = cv2.rectangle(detection_mask, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), 255, -1)
    return detection_mask
    
def yolo_detection_to_contour(segmented_frame, detection, other_detections=[]):
    """
    
    Arguments
    """
    
    detection_mask=detection2mask(detection, segmented_frame)
    
    and_mask = cv2.bitwise_and(segmented_frame, detection_mask)
    
    if other_detections:
        other_detections_masks=[detection2mask(detection_, segmented_frame) for detection_ in other_detections]
        for mask in other_detections_masks:
            and_mask = cv2.subtract(and_mask, mask)
    
    
    
    assert (and_mask == 255).any()
    
    contours = cv2.findContours(and_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    
    sorted(contours, key=lambda c: cv2.contourArea(c))
    contour = contours[-1]
        
    return contour

def yolo_detections_to_blobs(frame, segmented_frame, detections, **kwargs):
    """
    
    Arguments:
    
    * frame (np.ndarray):
    * detection (dict):
    * kwargs: Extra arguments to _create_blobs_objects
    """

    save_pixels="NONE"
    save_segmentation_image="NONE"
    contours = []
    indices = np.arange(len(detections)).tolist()

    for i, detection in enumerate(detections):
        other_detections = [detections[index] for k, index in enumerate(indices) if k != i]
        assert len(detections) == len(other_detections)+1
        contour = yolo_detection_to_contour(segmented_frame, detection, other_detections)
        contours.append(contour)
    
    (
        bounding_boxes,
        bounding_box_images,
        centroids,
        areas,
        pixels,
        estimated_body_lengths
    ) = _get_blobs_information_per_frame(
        frame,
        contours,
        save_pixels,
        save_segmentation_image
    )
        
    miniframes=[
        frame[
            bounding_box[0][1]:bounding_box[1][1],
            bounding_box[0][0]:bounding_box[1][0]
        ] for bounding_box in bounding_boxes
    ]
       
    blobs_in_frame = _create_blobs_objects(
        bounding_boxes,
        miniframes,
        centroids,
        areas,
        pixels,
        contours,
        estimated_body_lengths,
        save_segmentation_image=save_segmentation_image,
        save_pixels=save_pixels,
        **kwargs
    )
    
    return blobs_in_frame

def load_kwargs_for_blob_regeneration(store_path, video_object, chunk, frame_number, frame_idx):
    
    idtrackerai_folder = os.path.join(os.path.dirname(store_path), "idtrackerai")
    session_folder=os.path.sep.join(video_object.session_folder.split(os.path.sep)[2:])

    conf_file = os.path.join(os.path.dirname(store_path), os.path.basename(os.path.dirname(store_path)) + ".conf")

    with open(conf_file, "r") as filehandle:
        config = json.load(filehandle)

    episode_number = None
    for i, episode in enumerate(video_object.episodes_start_end):
        if episode[0] < frame_number < episode[1]:
            episode_number = i

    assert episode_number is not None
    episode_number

    config["resolution_reduction"]=config["_resreduct"]

    kwargs = {
        "video_path": video_object.video_path,
        "video_params_to_store": {"height": video_object.height, "width": video_object.width, "number_of_animals": video_object.user_defined_parameters["number_of_animals"]},
        "chunk": chunk,
        "global_frame_number": frame_number,
        "frame_number_in_video_path": frame_idx,
        "segmentation_parameters": config,
        "bounding_box_images_path": os.path.join(idtrackerai_folder, session_folder, f"segmentation_data/episode_images_{episode_number}.hdf5"),
        "pixels_path": os.path.join(idtrackerai_folder, session_folder, f"segmentation_data/episode_pixels_{episode_number}.hdf5")
    }
    return kwargs
    # assert os.path.exists(kwargs["bounding_box_images_path"])
    # assert os.path.exists(kwargs["pixels_path"])

def read_yolov7_label(label_file):
    with open(label_file, "r") as filehandle:
        lines = filehandle.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            print(line)

    return lines

def get_label_file_path(dataset, frame_number, chunk, frame_idx):
    labels_dir = f"runs/detect/{dataset}/labels/"
    labels_dir=os.path.join(yolov7_repo, labels_dir)

    key=f"{frame_number}_{chunk}-{frame_idx}"
    label_file = os.path.join(labels_dir, f"{key}.txt")

    assert os.path.exists(label_file)
    return label_file
