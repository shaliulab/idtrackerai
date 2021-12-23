import os.path
import logging

import imgstore
from .video import Video
from imgstore import new_for_filename

logger = logging.getLogger(__name__)

class VideoImgstore(Video):

    def __init__(self, store_list, chunk):

        self._stores = store_list
        self._chunk = chunk

        chunk_numbers = [chunk]

        self._store = new_for_filename(store_list[0], chunk_numbers=chunk_numbers)
        logger.info(f"Opening {self._store.current_video_path}")
        super().__init__(
            video_path = self._store.current_video_path,
            open_multiple_files=False
        )
