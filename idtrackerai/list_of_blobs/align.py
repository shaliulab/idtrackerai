import logging
import warnings
import tqdm
from confapp import conf


try:
    import imgstore.constants
    import imgstore.interface
    
    IMGSTORE_ENABLED=True
except ModuleNotFoundError:
    IMGSTORE_ENABLED=False

logger = logging.getLogger(__name__)


class AlignableList:
    
    def dealign(self, video_object):

        cap=imgstore.interface.VideoCapture(
            video_object.video_path,
            chunk=video_object._chunk
        )

        if getattr(conf, "MULTI_STORE_ENABLED", False):
            cap.select_store(conf.SELECTED_STORE)

            ids=cap.crossindex.get_refresh_ids()

            dealigned_blobs = []
            for id in tqdm.tqdm(ids, desc="Dealigning blobs"):
                dealigned_blobs.append(self.blobs_in_video[id[0]])

            self.blobs_in_video = dealigned_blobs

        else:
            pass
        
        self._annotate_location_of_blobs()


    def align(self, video_object):
        """
        This method takes the blobs_in_video and duplicates its elements
        (blobs_in_frame) so the number of blobs_in_frame matches the
        number of frames of the selected store
        (which is supposed to have a higher framerate)
        """


        cap=imgstore.interface.VideoCapture(
            video_object.video_path,
            chunk=video_object._chunk
        )

        if getattr(conf, "MULTI_STORE_ENABLED", False):
            cap.select_store(conf.SELECTED_STORE)
        
            cur = cap.crossindex._conn.cursor()
            cur.execute(f"SELECT COUNT(id) FROM selected;") # also from master is the same number :)
            # TODO This info should already be saved in a separate summary table
            nframes = cur.fetchone()[0]

            logger.info("Trimming delta index ")
            # we assume the frame_times are sorted in increasing manner
            # the indices indices are also sorted
                    
            logger.info("Done")

            aligned_blobs=[]
            warned=False
            
            rows = cap.crossindex.get_all_fn(cap.main)
            # for selected_fn in tqdm.tqdm(range(number_of_frames), desc="Aligning blobs"):
            for row in tqdm.tqdm(rows, desc="Aligning blobs"):
                try:
                    frame_number=row[0]
                    aligned_blobs.append(self.blobs_in_video[frame_number])
                except IndexError:
                    # TODO
                    # Figure out why the crossindex has a number of rows equal to frame_times (expected)
                    # but the highest frame number is the length of the original blobs in video,
                    # which is unexpected since it should be that - 1 (because the frame numbers are 0 indexed)
                    aligned_blobs.append([])
                    if not warned:
                        warnings.warn("Some frames from the index have been ignored", stacklevel=2)
                        warned=True

            assert len(aligned_blobs) == nframes
            self.blobs_in_video = aligned_blobs

        else:
            pass
