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

from idtrackerai.animals_detection.segmentation import _create_blobs_objects
from idtrackerai.animals_detection.segmentation_utils import (
    _get_blobs_information_per_frame,
    apply_segmentation_criteria
)

from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.utils.py_utils import read_json_file
from idtrackerai.animals_detection.yolov7.nms_multilabel import resolve_multilabel
from idtrackerai.animals_detection.yolov7.filter import keep_best_detections
from idtrackerai.utils.py_utils import download_file
from idtrackerai.animals_detection.yolov7.detection import Detection

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


def annotate_chunk_with_yolov7(store_path, session_folder, chunk, frames, input, allowed_classes=None, minimum_confidence=None, **kwargs):
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
        * input (str): Path to folder with labels (each frame must have a corresponding label here)
        * allowed_classes (iterable): id of the classes taken from the yolov7 output

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

    video_path=os.path.join(os.path.dirname(store_path), f"{str(chunk).zfill(6)}.mp4")
    assert os.path.exists(video_path)

    blobs_in_frame_all = []
    failed_frames = []
    successful_frames = []


    def record_failed_frame(fn):
        failed_frames.append(fn)
        logger.debug(f"YOLOv7 failed in frame %s", frame_number)

    cap = cv2.VideoCapture(video_path)
    last_frame_idx = None
    logger.info(f"YOLOv7 will annotate {len(frames)} frames")
    for frame_number, frame_idx in frames:
        if last_frame_idx is None or frame_idx != (last_frame_idx+1):
            cap.set(1, frame_idx)
        last_frame_idx = frame_idx
        assert cap.get(1) == frame_idx
        ret, frame = cap.read()
        frame=frame[:,:,0]

        label_file=get_label_file_path(input, frame_number, chunk, frame_idx)
        detections=load_detections_from_one_file(label_file, class_names=allowed_classes)
        
        
        kwargs_for_blob_regeneration=load_kwargs_for_blob_regeneration(store_path, video_object, chunk, frame_number, frame_idx)
        kwargs.update(kwargs_for_blob_regeneration)
        config=kwargs["segmentation_parameters"]
        roi=np.array(eval(config["_roi"]["value"][0][0]))
        mask = np.zeros_like(frame)
        mask = cv2.drawContours(mask, [roi], -1, 255, -1)
        detections=filter_detections_in_roi(detections, mask)

        if len(detections) > number_of_animals:
            # if two detections highly overlap, keep the blurry
            detections=resolve_multilabel(detections)
            if len(detections) == number_of_animals:
                logger.info(f"MULTILABEL RESOLVED %s", label_file)

            elif len(detections) > number_of_animals:
                # if still too many detections are found, keep the ones with the most confidence
                detections = keep_best_detections(detections, number_of_animals)
                logger.info(f"FILTERING detections in %s w confidence >= %s", label_file, detections[-1].conf)
            else:
                logger.info(f"NO DETECTIONS FOUND IN %s", label_file)


        if len(detections) == 0:
            record_failed_frame(frame_number)
            continue

        if allowed_classes is not None:
            detections=[detection for detection in detections if detection.class_id in allowed_classes]


        if minimum_confidence is not None:
            detections=[detection for detection in detections if detection.conf > minimum_confidence]


        if len(detections) == 0:
            record_failed_frame(frame_number)
            continue

        if len(detections) == video_object.user_defined_parameters["number_of_animals"]:


            segmented_frame, _ = apply_segmentation_criteria(frame, config)
            try:
                blobs_in_frame = yolo_detections_to_blobs(frame, config, segmented_frame, detections, frame_number=frame_number, **kwargs)
            except Exception as error:
                logger.error(error)
                logger.error(traceback.print_exc())
                print(f"Could not process YOLOv7 predictions in frame {frame_number}")
                record_failed_frame(frame_number)
                continue
            blobs_in_frame_all.append((frame_number, blobs_in_frame))
            successful_frames.append(frame_number)
        else:
            record_failed_frame(frame_number)
        # end of for loop over frames

    cap.release()
    step = "preprocessing"
    blobs_collection = os.path.join(session_folder, step, "blobs_collection.npy")
    assert os.path.exists(blobs_collection)
    try:
        shutil.copy(blobs_collection, f"{blobs_collection}.bak")
        list_of_blobs=ListOfBlobs.load(blobs_collection)

        if blobs_in_frame_all:
            for frame_number, blobs_in_frame in blobs_in_frame_all:
                list_of_blobs.apply_modification(frame_number, blobs_in_frame)


    except Exception as error:
        shutil.copy(f"{blobs_collection}.bak", blobs_collection)
        raise error

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
        detection.class_name=class_names[class_id]

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



    contours=cv2.findContours(and_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    contours=sorted(contours, key=lambda c: cv2.contourArea(c))
    contour=contours[-1]

    return contour

def yolo_detections_to_blobs(frame, config, segmented_frame, detections, exclusive=True, frame_number=None, **kwargs):
    """

    Arguments:

    * frame (np.ndarray):
    * detections (list):
    * kwargs: Extra arguments to _create_blobs_objects
    """

    save_pixels="DISK"
    save_segmentation_image="DISK"
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

def load_kwargs_for_blob_regeneration(store_path, video_object, chunk, frame_number, frame_idx):

    session_folder=video_object.session_folder
    conf_file = os.path.join(os.path.dirname(store_path), os.path.basename(os.path.dirname(store_path)) + ".conf")
    config = read_json_file(conf_file)

    episode_number = None
    for i, episode in enumerate(video_object.episodes_start_end):
        if episode[0] <= frame_number < episode[1]:
            episode_number = i

    assert episode_number is not None
    episode_number

    config["resolution_reduction"]=config["_resreduct"]["value"]

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

    assert os.path.exists(kwargs["bounding_box_images_path"])
    assert os.path.exists(kwargs["pixels_path"])

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
