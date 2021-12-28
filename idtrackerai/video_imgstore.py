import os.path
import logging

import imgstore
from .video import Video
from imgstore import new_for_filename

logger = logging.getLogger(__name__)

class VideoImgstore(Video):

    def __init__(self, store, chunk=0):

        self._chunk = chunk
        chunk_numbers = [chunk]

        self._store = new_for_filename(
            store,
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
        d["_store"] = new_for_filename(
            full_path,
            [d["_chunk"]]
        )
        return super().__setstate__(d)