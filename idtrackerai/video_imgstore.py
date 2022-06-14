import os.path
import logging

import imgstore
from .video import Video
from imgstore.stores import multi as imgstore

logger = logging.getLogger(__name__)

from confapp import conf
try:
    import local_settings

    conf += local_settings
    logger.info(
        f"""        
    Using {conf.NUMBER_OF_JOBS_FOR_CONNECTING_BLOBS} jobs for parallel blob connection
    """
    )

except ImportError:
    pass


class VideoImgstore(Video):

    def __init__(self, store, chunk=0, init_store=True):

        self._chunk = chunk
        chunk_numbers = [chunk-1, chunk, chunk+1]
        chunk_numbers = [c for c in chunk_numbers if c >= 0]
        self._chunk_numbers = chunk_numbers
        self._store_path = store

        if init_store:
            self._store = imgstore.new_for_filename(store)
        else:
            self._store = None

        logger.info(f"Opening {self._store.current_video_path}")
        super().__init__(
            video_path = self._store.current_video_path,
            open_multiple_files=False
        )

    @property
    def store(self):
        if self._store is None:
            self._store = imgstore.new_for_filename(self._store_path)


    def __getstate__(self):
        d = self.__dict__.copy()
        store = d.pop("_store")
        d["full_path"] = store.full_path
        return d

    def __setstate__(self, d):

        # compatibility
        full_path = d.pop("full_path", None)
        if full_path is None:
            full_path = d.pop("_store_path")
        d["_store_path"] = full_path
        ####

        if "init_store" not in d:
            d["init_store"] = False


        relink = conf.WRONG_ROOT_DIR is not None and conf.RIGHT_ROOT_DIR is not None

        if relink:
            # path to the .yaml
            full_path = full_path.replace(
                conf.WRONG_ROOT_DIR,
                conf.RIGHT_ROOT_DIR
            )
            # path to the .npy
            d["_path_to_video_object"] = d["_path_to_video_object"].replace(
                conf.WRONG_ROOT_DIR,
                conf.RIGHT_ROOT_DIR
            )
            d["_session_folder"] = d["_session_folder"].replace(
                conf.WRONG_ROOT_DIR,
                conf.RIGHT_ROOT_DIR
            )
            # path to .mp4
            d["_video_path"] = d["_video_path"].replace(
                conf.WRONG_ROOT_DIR,
                conf.RIGHT_ROOT_DIR
            )

        self.__dict__ = d

        if relink:
            print(f"Saving relinked npy file to {self.path_to_video_object}")
            self.save()


    def get_first_frame(self, *args, **kwargs):
        self._store.reset_to_first_frame()
        return self._store._chunk_md["frame_number"][0]
