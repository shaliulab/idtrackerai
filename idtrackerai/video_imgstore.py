import os.path
import logging

import imgstore
from .video import Video
from imgstore.multistores import MultiStore

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

    def __init__(self, store, chunk=0):

        self._chunk = chunk
        chunk_numbers = [chunk-1, chunk, chunk+1]
        chunk_numbers = [c for c in chunk_numbers if c >= 0]
        self._chunk_numbers = chunk_numbers

        self._store = MultiStore.new_for_filename(
            store,
            ref_chunk=chunk,
            chunk_numbers=chunk_numbers
        )

        logger.info(f"Opening {self._store.current_video_path}")
        super().__init__(
            video_path = self._store.current_video_path,
            open_multiple_files=False
        )


    def __getstate__(self):
        d = self.__dict__.copy()
        store = d.pop("_store")
        d["full_path"] = store.full_path
        return d

    def __setstate__(self, d):
        full_path = d.pop("full_path")
        original_path = full_path
        reset = conf.WRONG_ROOT_DIR is not None and conf.RIGHT_ROOT_DIR is not None

        if reset:
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


        d["_store"] = MultiStore.new_for_filename(
            full_path,
            ref_chunk=d["_chunk"],
            chunk_numbers=d["_chunk_numbers"]
        )
        self.__dict__ = d

        if reset:

            print(f"Saving relinked npy file to {self.path_to_video_object}")
            self.save()


    def get_first_frame(self, *args, **kwargs):
        self._store.reset_to_first_frame()
        return self._store._chunk_md["frame_number"][0]
