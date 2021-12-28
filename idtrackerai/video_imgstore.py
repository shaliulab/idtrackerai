import os.path
import logging

import imgstore
from .video import Video
from imgstore.multistores import MultiStore

logger = logging.getLogger(__name__)

class VideoImgstore(Video):

    def __init__(self, store, chunk=0):

        self._chunk = chunk
        chunk_numbers = [chunk-1, chunk, chunk+1]
        chunk_numbers = [c for c in chunk_numbers if c >= 0]
        self._chunk_numbers = chunk_numbers

        self._store = MultiStore.new_for_filename(
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
        d["_store"] = MultiStore.new_for_filename(
            full_path,
            chunk_numbers=d["_chunk_numbers"]
        )
        self.__dict__ = d

        #return super(VideoImgstore, self).__setstate__(d)

    def get_first_frame(self, *args, **kwargs):
        self._store.reset_to_first_frame()
        return self._store._chunk_md["frame_number"][0]
