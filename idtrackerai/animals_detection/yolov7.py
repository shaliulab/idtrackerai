import os.path
import logging
import traceback
import shutil
import cv2
import numpy as np

from .segmentation import _create_blobs_objects
from .segmentation_utils import (
    _get_blobs_information_per_frame,
    apply_segmentation_criteria
)

from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.utils.py_utils import read_json_file

logger=logging.getLogger(__name__)

yolov7_repo = "/scratch/leuven/333/vsc33399/Projects/YOLOv7/yolov7"

def annotate_chunk_with_yolov7(store_path, chunk, frames, allowed_classes=None, save=True, **kwargs):
    """
    Correct idtrackerai preprocessing errors with YOLOv7 results,
    which should be made available in the runs/detect folder of the YOLOv7 repository
    under a folder called like the experiment
    
    The function loads the list_of_blobs,
    modifies it with the information collected from the YOLOv7 output
    and saves it back to the original file
    
    The attribute yolov7_annotated of the list_of_blobs contains the frame number
    of the frames where a modification has been made

    Arguments:
    
        * store_path (str): Path to the metadata.yaml file of the imgstore
        * chunk (int): Chunk to be processed
        * frames (list): frame_number, frame_idx tuples for each frame in the experiment where idtrackerai preprocessing failed
        * allowed_classes (iterable): id of the classes taken from the yolov7 output
        
    Returns:
        None
    """

    assert len(frames) > 0
    
    idtrackerai_folder=os.path.join(os.path.dirname(store_path), "idtrackerai")
    video_object_path = os.path.join(idtrackerai_folder, f"session_{str(chunk).zfill(6)}", "video_object.npy")
    assert os.path.exists(video_object_path)
    video_object = np.load(video_object_path, allow_pickle=True).item()
    session_folder=os.path.sep.join(video_object.session_folder.split(os.path.sep)[2:])

    # TODO Let's get a more robust way of getting the experiment
    # maybe as a property of the metadata
    experiment = os.path.sep.join(
        os.path.dirname(store_path).split(os.path.sep)[-3:]
    )
    
    # get the problematic frame
    video_path=os.path.join(os.path.dirname(store_path), f"{str(chunk).zfill(6)}.mp4")
    assert os.path.exists(video_path)
    
    blobs_in_frame_all = []

    cap = cv2.VideoCapture(video_path)
    for frame_number, frame_idx in frames:
        cap.set(1, frame_idx)
        ret, frame = cap.read()
        frame=frame[:,:,0]
    
        label_file=get_label_file_path(experiment, frame_number, chunk, frame_idx)
        lines=read_yolov7_label(label_file)
        detections = [yolo_line_to_detection(line) for line in lines]

        if len(detections) == 0:
            print(f"No detections found for frame_number {frame_number}")
            logger.debug(f"Label file {label_file}")
            continue

        if allowed_classes is not None:
            detections=[detection for detection in detections if detection["class"] in allowed_classes]

        if len(detections) == 0:
            print(f"No detections found for frame_number {frame_number} for allowed classes {allowed_classes}")
            logger.debug(f"Label file {label_file}")
            continue


        kwargs.update(load_kwargs_for_blob_regeneration(store_path, video_object, chunk, frame_number, frame_idx))
        config=kwargs["segmentation_parameters"]
        
        if len(detections) == video_object.user_defined_parameters["number_of_animals"]:
            segmented_frame, _ = apply_segmentation_criteria(frame, config)
            try:
                blobs_in_frame = yolo_detections_to_blobs(frame, segmented_frame, detections, frame_number=frame_number, **kwargs)
            except Exception as error:
                logger.error(error)
                logger.error(traceback.print_exc())
                print(f"Could not process YOLOv7 predictions in frame {frame_number}")
                continue
            blobs_in_frame_all.append((frame_number, blobs_in_frame))
        else:
            print(f"YOLOv7 failed in frame {frame_number}")
        
    cap.release()

    blobs_collection = os.path.join(idtrackerai_folder, session_folder, "preprocessing", "blobs_collection.npy")
    assert os.path.exists(blobs_collection)
    try:
        shutil.copy(blobs_collection, f"{blobs_collection}.bak")
        list_of_blobs=ListOfBlobs.load(blobs_collection)
        
        if blobs_in_frame_all:
            for frame_number, blobs_in_frame in blobs_in_frame_all:
                list_of_blobs.apply_modification(frame_number, blobs_in_frame)
        
        if save:
            list_of_blobs.save(blobs_collection)
    except Exception as error:
        shutil.copy(f"{blobs_collection}.bak", blobs_collection)
        raise error
    
    success_rate = round(len(blobs_in_frame_all) / len(frames), 3)
    return list_of_blobs, success_rate



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
    
    contours=cv2.findContours(and_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    contours=sorted(contours, key=lambda c: cv2.contourArea(c))
    contour=contours[-1]
        
    return contour

def yolo_detections_to_blobs(frame, segmented_frame, detections, exclusive=True, frame_number=None, **kwargs):
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
        if exclusive:
            other_detections = [detections[index] for k, index in enumerate(indices) if k != i]
            assert len(detections) == len(other_detections)+1
        else:
            other_detections = []

        contour = yolo_detection_to_contour(segmented_frame, detection, other_detections)
        assert contour.shape[0] >= 4
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
        save_segmentation_image,
        tight=True,
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
    for blob in blobs_in_frame:
        blob.modified=True
        blob.segmentation_contour = blob.contour.copy()
        bbox=blob.bounding_box_in_frame_coordinates
        bbox = [bbox[0][0], bbox[0][1], bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1]]
        blob.contour = np.array([
            [bbox[0], bbox[1]],
            [bbox[0]+bbox[2], bbox[1]],
            [bbox[0]+bbox[2], bbox[1]+bbox[3]],
            [bbox[0], bbox[1]+bbox[3]],
        ]).reshape((4, 1, 2))
    
    return blobs_in_frame

def load_kwargs_for_blob_regeneration(store_path, video_object, chunk, frame_number, frame_idx):
    
    idtrackerai_folder = os.path.join(os.path.dirname(store_path), "idtrackerai")
    session_folder=os.path.sep.join(video_object.session_folder.split(os.path.sep)[2:])

    conf_file = os.path.join(os.path.dirname(store_path), os.path.basename(os.path.dirname(store_path)) + ".conf")
    config = read_json_file(conf_file)

    episode_number = None
    for i, episode in enumerate(video_object.episodes_start_end):
        if episode[0] <= frame_number < episode[1]:
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
    return lines

def get_label_file_path(dataset, frame_number, chunk, frame_idx):
    labels_dir = f"runs/detect/{dataset}/incomplete_frames/yolov7/labels/"
    labels_dir=os.path.join(yolov7_repo, labels_dir)

    key=f"{frame_number}_{chunk}-{frame_idx}"
    label_file = os.path.join(labels_dir, f"{key}.txt")

    assert os.path.exists(label_file), f"{label_file} cannot be found"
    return label_file
