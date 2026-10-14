import os.path
import logging
import traceback
import shutil
import math
import tempfile
import itertools

from tqdm.auto import tqdm
import cv2
import numpy as np
import joblib

from idtrackerai.animals_detection.segmentation import _create_blobs_objects, get_segmentation_storage
from idtrackerai.animals_detection.segmentation_utils import (
    _get_blobs_information_per_frame,
    apply_segmentation_criteria,
    get_roi_mask,
    parse_contours,
)

from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.utils.py_utils import read_json_file
from idtrackerai.animals_detection.yolov7.nms_multilabel import resolve_multilabel
from idtrackerai.animals_detection.yolov7.filter import keep_best_detections
from idtrackerai.utils.py_utils import download_file
from idtrackerai.animals_detection.yolov7.detection import Detection

try:
    from imgstore.interface import VideoCapture
except ModuleNotFoundError:
    from cv2 import VideoCapture

logger=logging.getLogger(__name__)

yolov7_repo = "/scratch/leuven/333/vsc33399/Projects/YOLOv7/yolov7"

def filter_detections_in_roi(detections, mask):

    detections_in_roi=[]
    for detection in detections:

        bbox = (
                int(detection.x * mask.shape[1]),
                int(detection.y * mask.shape[0]),
                int(detection.w * mask.shape[1]),
                int(detection.h * mask.shape[0]),
        )
        if np.any(mask[
            bbox[1]:(bbox[1]+bbox[3]),
            bbox[0]:(bbox[0]+bbox[2])
            ]):

            detections_in_roi.append(detection)
    return detections_in_roi


def read_frames(store_path, chunk, frame_numbers):
    """
    Yield (frame_number, frame) for each of the passed frame numbers of the chunk, in increasing order,
    read with the same reader used during preprocessing, so the frames are identical (same cropping)

    The frame is None if it could not be read
    """
    cap = VideoCapture(store_path, chunk)
    try:
        last_frame_number = None
        for frame_number in sorted(frame_numbers):
            # only seek if the frames are not consecutive
            if last_frame_number is None or frame_number != last_frame_number + 1:
                cap.set(1, frame_number)
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.error("Could not read frame %s of chunk %s", frame_number, chunk)
                last_frame_number = None
                yield frame_number, None
                continue
            last_frame_number = frame_number
            if frame.ndim == 3:
                frame = frame[:,:,0]
            yield frame_number, frame
    finally:
        cap.release()


def annotate_chunk_with_yolov7(store_path, session_folder, chunk, frames, input, allowed_classes=None, minimum_confidence=None, list_of_blobs=None, **kwargs):
    """
    Correct idtrackerai preprocessing errors with YOLOv7 results,
    which should be made available in the runs/detect folder of the YOLOv7 repository
    under a folder called like the experiment

    The function modifies the list_of_blobs (loaded from the preprocessing step if not passed)
    with the information collected from the YOLOv7 output

    The attribute yolov7_annotated of the list_of_blobs contains the frame number
    of the frames where a modification has been made

    Arguments:

        * store_path (str): Path to the metadata.yaml file of the imgstore
        * chunk (int): Chunk to be processed
        * frames (list): frame_number, frame_idx tuples for each frame in the experiment where idtrackerai preprocessing failed
        * input (str): Path to folder with labels (each frame must have a corresponding label here)
        * allowed_classes (dict): id -> name of the classes taken from the yolov7 output
        * minimum_confidence (float): If passed, detections with lower confidence are ignored
        * list_of_blobs (ListOfBlobs): blobs produced by the preprocessing step

    Returns:
        * list_of_blobs
        * successful_frames (list): Frame number of frames where the number of objects of the allowed class matches the expected.
        which is the number of animals recorded in the video_object of the chunk
        * failed_frames (list): Frame numbers where this condition is not met
    """

    assert len(frames) > 0


    video_object_path = os.path.join(session_folder, "video_object.npy")
    assert os.path.exists(video_object_path)
    video_object = np.load(video_object_path, allow_pickle=True).item()
    number_of_animals=video_object._user_defined_parameters["number_of_animals"]

    if list_of_blobs is None:
        blobs_collection = os.path.join(session_folder, "preprocessing", "blobs_collection.npy")
        assert os.path.exists(blobs_collection)
        list_of_blobs=ListOfBlobs.load(blobs_collection)

    blobs_in_frame_all = []
    failed_frames = []
    successful_frames = []

    def record_failed_frame(fn):
        failed_frames.append(fn)
        logger.debug(f"YOLOv7 failed in frame %s", fn)

    # these do not change from frame to frame
    config = load_config(store_path)
    save_pixels, save_segmentation_image = get_segmentation_storage(number_of_animals)
    if save_pixels == "NONE":
        # the contour of the blobs modified by YOLOv7 is replaced by its bounding box,
        # so their pixels cannot be recomputed from the contour
        save_pixels = "RAM"

    frame_idxs = dict(frames)
    roi_mask = None

    logger.info(f"YOLOv7 will annotate {len(frames)} frames")
    for frame_number, frame in read_frames(store_path, chunk, frame_idxs.keys()):
        frame_idx = frame_idxs[frame_number]
        if frame is None:
            record_failed_frame(frame_number)
            continue

        if frame.shape != (video_object.height, video_object.width):
            raise ValueError(
                f"Frame {frame_number} has shape {frame.shape}, "
                f"but the video was preprocessed with shape {(video_object.height, video_object.width)}"
            )

        if roi_mask is None:
            roi_mask = get_roi_mask(config, frame.shape)

        label_file=get_label_file_path(input, frame_number, chunk, frame_idx)
        detections=load_detections_from_one_file(label_file, class_names=allowed_classes)

        # filter by class and confidence before choosing the best detections
        if allowed_classes is not None:
            detections=[detection for detection in detections if detection.class_id in allowed_classes]

        if minimum_confidence is not None:
            detections=[detection for detection in detections if detection.conf > minimum_confidence]

        detections=filter_detections_in_roi(detections, roi_mask)

        if len(detections) > number_of_animals:
            # if two detections highly overlap, keep the blurry
            detections=resolve_multilabel(detections)
            if len(detections) == number_of_animals:
                logger.info(f"MULTILABEL RESOLVED %s", label_file)

            elif len(detections) > number_of_animals:
                # if still too many detections are found, keep the ones with the most confidence
                detections = keep_best_detections(detections, number_of_animals)
                logger.info(f"FILTERING detections in %s w confidence >= %s", label_file, detections[-1].conf)

        if len(detections) == number_of_animals:

            blob_kwargs = get_kwargs_for_blob_regeneration(store_path, video_object, chunk, frame_number, config)
            segmented_frame, _ = apply_segmentation_criteria(frame, config, roi_mask=roi_mask)
            try:
                blobs_in_frame = yolo_detections_to_blobs(
                    frame, config, segmented_frame, detections, frame_number=frame_number,
                    save_pixels=save_pixels, save_segmentation_image=save_segmentation_image,
                    **blob_kwargs, **kwargs
                )
            except Exception:
                logger.exception(f"Could not process YOLOv7 predictions in frame {frame_number}")
                record_failed_frame(frame_number)
                continue
            blobs_in_frame_all.append((frame_number, blobs_in_frame))
            successful_frames.append(frame_number)
        else:
            record_failed_frame(frame_number)
        # end of for loop over frames

    for frame_number, blobs_in_frame in blobs_in_frame_all:
        list_of_blobs.apply_modification(frame_number, blobs_in_frame)

    return list_of_blobs, successful_frames, failed_frames


def load_detections_from_one_file(label_file, count=None, class_id=None, false_action=None, true_action=None, class_names={}):

    lines=read_yolov7_label(label_file)
    detections = [yolo_line_to_detection(line, class_names) for line in lines]
    result=detections
    experiment="_".join(label_file.split("idtrackerai")[0].split(os.path.sep)[-3:])


    if class_id is not None:
        detections = [detection for detection in detections if detection.class_id == class_id]

    if count is not None:
        detection_count = len(detections)
        if detection_count != count:
            result=None

    if result is None:
        if false_action:
            false_action(label_file, detections=detections, experiment=experiment)
    else:
        if true_action:
            true_action(label_file, detections=detections, experiment=experiment)

    if false_action or true_action:
        return None
    else:
        return result

def load_detections_from_files(label_files, **kwargs):

    detections = []
    for label_file in tqdm(label_files, desc="Loading detections"):
        detections.append(load_detections_from_one_file(label_file, **kwargs))

    return detections

def load_detections(label_files, n_jobs=1,  **kwargs):

    if n_jobs > 0:
        jobs = n_jobs
    else:
        nproc = os.sched_getaffinity(0)
        jobs = max(1, nproc - n_jobs)

    partition_size = math.ceil(len(label_files) / jobs)

    label_files_partition = [
        label_files[(i*partition_size):((i+1)*partition_size)]
        for i in range(jobs)
    ]

    detections = joblib.Parallel(n_jobs=jobs)(
        joblib.delayed(load_detections_from_files)(files, **kwargs)
        for files in label_files_partition
    )

    detections = list(itertools.chain(*detections))

    assert len(detections) == len(label_files)

    return detections




def yolo_line_to_detection(line, class_names={}):
    """

    Argumennts:

    * line (str): Label as read from a yolo .txt file

    Returns

    * detection (dict): class (int), bounding_box (x, y, w, h) and confidence (0 to 1)
    """

    line=line.split(" ")

    # class id
    class_id = int(line[0])

    # bounding_box and confidence optinally
    for i in range(1, len(line)):
        line[i] = float(line[i])

    bounding_box = (line[1]-line[3]/2, line[2]-line[4]/2, *line[3:5])
    x,y,w,h=bounding_box


    detection=Detection(class_id=class_id, conf=float(line[5]), x=x, y=y, w=w, h=h)


    if class_names:
        detection.class_name=class_names.get(class_id)

    return detection

def detection2mask(detection, frame):
    bbox_norm = detection.bounding_box
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

def yolo_detection_to_contour(frame, config, segmented_frame, detection, other_detections=[]):
    """

    Arguments
    """

    detection_mask=detection2mask(detection, segmented_frame)

    and_mask = cv2.bitwise_and(segmented_frame, detection_mask)

    if other_detections:
        other_detections_masks=[detection2mask(detection_, segmented_frame) for detection_ in other_detections]
        for mask in other_detections_masks:
            and_mask = cv2.subtract(and_mask, mask)


    if not (and_mask == 255).any():
        assert detection.class_name is not None
        if detection.class_name == "fly":
            raise ValueError("Animal not found")
        else:
            and_mask=detection_mask



    contours=parse_contours(cv2.findContours(and_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE))
    contours=sorted(contours, key=lambda c: cv2.contourArea(c))
    contour=contours[-1]

    return contour

def yolo_detections_to_blobs(frame, config, segmented_frame, detections, exclusive=True, frame_number=None, save_pixels="DISK", save_segmentation_image="DISK", **kwargs):
    """
    Process the yolov7 detections into idtrackerai blobs

    Arguments:

    * frame (np.ndarray): Original frame
    * detections (list): Collection of idtrackerai.animals_detection.yolov7.detection.Detection objects
    * kwargs: Extra arguments to _create_blobs_objects
    """

    contours = []
    indices = np.arange(len(detections)).tolist()

    for i, detection in enumerate(detections):
        if exclusive:
            other_detections = [detections[index] for k, index in enumerate(indices) if k != i]
            assert len(detections) == len(other_detections)+1
        else:
            other_detections = []

        contour = yolo_detection_to_contour(frame, config, segmented_frame, detection, other_detections)
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
        modified=True,
        **kwargs
    )
    for i, blob in enumerate(blobs_in_frame):
        blob.modified=True
        blob._annotation["class"]=detections[i].class_name
        blob._annotation["confidence"]=detections[i].conf
        blob.segmentation_contour = blob.contour.copy()
        bbox=blob.bounding_box_in_frame_coordinates
        bbox = [bbox[0][0], bbox[0][1], bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1]]
        blob.contour = np.array([
            [bbox[0], bbox[1]],
            [bbox[0]+bbox[2]-1, bbox[1]],
            [bbox[0]+bbox[2]-1, bbox[1]+bbox[3]-1],
            [bbox[0], bbox[1]+bbox[3]-1],
        ]).reshape((4, 1, 2))

    return blobs_in_frame

def load_config(store_path):
    """Load the .conf file of the experiment (next to the metadata.yaml) with the resolution reduction"""
    conf_file = os.path.join(os.path.dirname(store_path), os.path.basename(os.path.dirname(store_path)) + ".conf")
    config = read_json_file(conf_file)
    config["resolution_reduction"]=config["_resreduct"]["value"]
    return config


def get_kwargs_for_blob_regeneration(store_path, video_object, chunk, frame_number, config):

    session_folder=video_object.session_folder

    episode_number = None
    for i, episode in enumerate(video_object.episodes_start_end):
        if episode[0] <= frame_number < episode[1]:
            episode_number = i
            break
    assert episode_number is not None, f"Frame {frame_number} is not in any episode"

    kwargs = {
        "video_path": video_object.video_path,
        "video_params_to_store": {"height": video_object.height, "width": video_object.width, "number_of_animals": video_object.user_defined_parameters["number_of_animals"]},
        "chunk": chunk,
        "global_frame_number": frame_number,
        "frame_number_in_video_path": frame_number, # this would be the frame idx in the multiple video interface from idtrackerai, but not if we use imgstore
        "segmentation_parameters": config,
        "bounding_box_images_path": os.path.join(session_folder, f"segmentation_data/episode_images_{episode_number}.hdf5"),
        "pixels_path": os.path.join(session_folder, f"segmentation_data/episode_pixels_{episode_number}.hdf5")
    }
    return kwargs


def load_kwargs_for_blob_regeneration(store_path, video_object, chunk, frame_number, frame_idx):
    return get_kwargs_for_blob_regeneration(store_path, video_object, chunk, frame_number, load_config(store_path))


def read_yolov7_label(label_file):

    if label_file.startswith("http://"):
        extension = os.path.splitext(label_file)[1]
        temp_file = tempfile.mktemp(prefix="yolov7", suffix=extension)
        download_file(label_file, temp_file)
        label_file = temp_file

    with open(label_file, "r") as filehandle:
        lines = filehandle.readlines()
        lines = [line.strip() for line in lines]
    return lines

def get_label_file_path(labels_dir, frame_number, chunk, frame_idx):

    key=f"{frame_number}_{chunk}-{frame_idx}"
    label_file = os.path.join(labels_dir, f"{key}.txt")

    assert os.path.exists(label_file), f"{label_file} cannot be found"
    return label_file
