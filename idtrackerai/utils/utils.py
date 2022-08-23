import argparse
from confapp import conf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from idtrackerai.animals_detection import AnimalsDetectionAPI 
detection_parameters_keys=AnimalsDetectionAPI.detection_parameters_keys

def load_mask_img(store, config):
    roi = np.array(eval("[" + config["_roi"]["value"][0][0] + "]"))
    mask=store.get_black_mask()
    mask=cv2.drawContours(mask, [roi], -1, 255, -1)
    return mask


def load_segmentation_parameters(store, config):
    """
    Restore the segmentation parameters object, as used in idtracker.ai runs
    
    Arguments:
        * store (imgstore.VideoStore): Instance of an imgstore being analyzed with idtrackerai
        * config (dict): Contents of the config.conf file passed to idtrackerai
        * frame_number (int): id (frame number) inside the store 
        * idx (int): Index of the feed to be processed, as output by the imgstore
        
    Return:
        * segmentation_parameters (dict)
    """
    
    data={}
    for k, v in config.items():   
        if "__iter__" in dir(v):
            value = {}
            for kk in v:
                if v[kk] == "False":
                    value[kk] = False
                elif v[kk] == "True":
                    value[kk] = True
                else:
                    value[kk] = v[kk] 
            prop = argparse.Namespace(**value)
        else:
            prop = v

        data[k] = prop

    self = argparse.Namespace(**data)

    self._tracking_interval = self._range.value
    self._mask_img = load_mask_img(store, config)
    self._background_img = None


    user_defined_parameters = {
                "number_of_animals": int(self._number_of_animals.value),
                "min_threshold": self._intensity.value[0],
                "max_threshold": self._intensity.value[1],
                "min_area": self._area.value[0],
                "max_area": self._area.value[1],
                "check_segmentation": self._chcksegm.value,
                "tracking_interval": self._tracking_interval,
                "apply_ROI": self._applyroi.value,
                "rois": self._roi.value,
                "mask": self._mask_img,
                "subtract_bkg": self._bgsub.value,
                "bkg_model": self._background_img,
                "resolution_reduction": self._resreduct.value,
                "track_wo_identification": self._no_ids.value,
                "setup_points": None,
                "sigma_gaussian_blurring": conf.SIGMA_GAUSSIAN_BLURRING,
                "knowledge_transfer_folder": conf.KNOWLEDGE_TRANSFER_FOLDER_IDCNN,
                "identity_transfer": False,
                "identification_image_size": None,
            }
    segmentation_parameters = {k: user_defined_parameters[k] for k in detection_parameters_keys}
    return segmentation_parameters