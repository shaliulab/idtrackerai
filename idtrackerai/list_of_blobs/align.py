import logging
import warnings
import tqdm
from confapp import conf, load_config


try:
    import imgstore.constants
    import imgstore.interface
    
    IMGSTORE_ENABLED=True
except ModuleNotFoundError:
    IMGSTORE_ENABLED=False

logger = logging.getLogger(__name__)


def find_frame_number(store, ft):
    """
    Given a frame time, return the frame number
    of the first frame after that in the index
    """
    # NOTE:
    # This is more efficient than the commented code below because
    # this skips the "ORDER BY" SQL statement, which is very slow
    # In this way, we dont need the ORDER BY statement because we take
    # advantage of the fact that the frame times are already ordered
    
    # Here we look for the first frame_number in the master store that is ahead of the passed frame time
    # (from the selected or delta store) and we subtract one from it
    # to get the first one in the past instead of the future
    cur = store._conn.cursor()
    cmd="SELECT id from master WHERE (frame_time - ?) >= 0 LIMIT 1;"
    cur.execute(cmd, (ft,))
    frame_number = cur.fetchone()[0]
    # TODO Check data
    return frame_number


class AlignableList:
    
    def dealign(self, video_object):

        config = load_config(imgstore.constants)
        cap=imgstore.interface.VideoCapture(
            video_object.video_path,
            chunk=video_object._chunk
        )

        if getattr(config, "MULTI_STORE_ENABLED", False):
            cap.select_store(config.SELECTED_STORE) 

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


        config = load_config(imgstore.constants)
        cap=imgstore.interface.VideoCapture(
            video_object.video_path,
            chunk=video_object._chunk
        )

        if getattr(config, "MULTI_STORE_ENABLED", False):
            cap.select_store(config.SELECTED_STORE)            
        
            cur = cap.crossindex._conn.cursor()
            cur.execute(f"SELECT COUNT(id) FROM selected;") # also from master is the same number :)
            # TODO This info should already be saved in a separate summary table
            nframes = cur.fetchone()[0]

            index=cap._master._index

            logger.info("Trimming delta index ")
            # we assume the frame_times are sorted in increasing manner
            # the indices indices are also sorted
                    
            logger.info("Done")

            # start_of_data = find_frame_number(cap, index.get_chunk_metadata(video_object._chunk)["frame_time"][0])
            # end_of_data = find_frame_number(cap, index.get_chunk_metadata(video_object._chunk)["frame_time"][-1])
            # print(f"Starting frame: {start_of_data}, Ending frame {end_of_data}")

            aligned_blobs=[]
            warned=False
            
            rows = cap.crossindex.get_all_master_fn()
            # for selected_fn in tqdm.tqdm(range(number_of_frames), desc="Aligning blobs"):
            for row in tqdm.tqdm(rows, desc="Aligning blobs"):
                try:
                    # master_fn = cap.crossindex.find_master_fn(selected_fn)
                    master_fn=row[0]
                    aligned_blobs.append(self.blobs_in_video[master_fn])
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