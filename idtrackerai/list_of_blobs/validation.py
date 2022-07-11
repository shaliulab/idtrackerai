import os.path
import pandas as pd
import numpy as np

# VALIDATION_FILE is a csv with at least a column called id
# listing frame_numbers which need validation
VALIDATION_FILE="frames-to-validate.csv"
if os.path.exists(VALIDATION_FILE):
    frames_to_validate = pd.read_csv(VALIDATION_FILE, index_col=0)
else:
    VALIDATION_FILE=None
    frames_to_validate = None

# TODO: consider moving to validation
def check_tracking(blobs_in_frame, number_of_animals):
    """Returns True if the list of blobs `blobs_in_frame` needs to be
    validated.

    A list of blobs of a frame need to be validated if some blobs are crossings
    or if there is some missing identity.

    Parameters
    ----------
    blobs_in_frame : list
        List of Blob objects in a given frame of the video.

    Returns
    -------
    check_tracking_flag : boolean
    """
    there_are_crossings = any(
        [blob.is_a_crossing and blob.user_generated_identities is None for blob in blobs_in_frame]
    )  # check whether there is a crossing in the frame
    missing_identity = any(
        [
            None in blob.final_identities or 0 in blob.final_identities
            for blob in blobs_in_frame
        ]
    )  # Check whether there is some missing identities (0 or None)

    unicity_cond = len(blobs_in_frame) == number_of_animals
    return there_are_crossings or missing_identity or (not unicity_cond)


def validate_from_file(blobs_in_video, current_frame, direction, number_of_animals, must_validate):

    if frames_to_validate is None:
        return None
    
    frame_numbers = frames_to_validate["id"]
    offset = frame_numbers - current_frame
    current_position = np.where(offset > 0)[0][0] - 1
    
    if direction == "future":
        try:
            fn=frame_numbers[current_position+1]
        except Exception as error:
            print(error)
            return None
        next = +1
    elif direction == "past":
        fn=frame_numbers[current_position]
        next = -1

    blobs_in_frame=blobs_in_video[fn]
    target = check_tracking(blobs_in_frame, number_of_animals)
    if not must_validate:
        target = not target

    while not target:
        fn += next
        try:
            blobs_in_frame=blobs_in_video[fn]
        except IndexError:
            fn = None
            break

    return fn

