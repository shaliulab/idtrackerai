import time
import multiprocessing
import logging
import math
from tqdm import tqdm
from confapp import conf

from .overlap import compute_overlapping_between_two_subsequent_frames

logger = logging.getLogger("__main__.list_of_blobs.parallel")

ROUND_FACTOR=1000
n_jobs=conf.NUMBER_OF_JOBS_FOR_CONNECTING_BLOBS

def compute_overlapping_between_subsequent_frames_single_job(
    blobs_in_video, f, start_and_end=None, queue=None
):

    """
    Arguments:

        start (tuple): Index of the original blobs_in_video from which this job will process
        Only used in the progress bar, so not essential
    """

    data = []

    if start_and_end is None:
        desc="Connecting blobs "
    else:
        desc=f"Connecting blobs from {start_and_end[0]} to {start_and_end[1]} "

    for frame_i in tqdm(
        range(1, len(blobs_in_video)), desc=desc
    ):
        data.extend(f(blobs_in_video[frame_i - 1], blobs_in_video[frame_i], queue, do=False))

    return data


class ParallelBlobOverlap:
    """
    A mixin class that implements the parallel processing of blobs in a video
    making full use of the computer's CPU (instead of just 1) 
    """

    @staticmethod
    def make_process_name(start, end):
        return f"Process [{start} - {end})"

    def fetch_blob(self, frame_identifier, blob_identifier):
        try:
            blobs_prior = self.blobs_in_video[frame_identifier]
        except Exception:
            raise Exception(f"Cannot fetch frame {frame_identifier} from blobs_in_video of length {len(self.blobs_in_video)}")
        else:
            try:
                blob = blobs_prior[blob_identifier]
            except Exception:
                raise Exception(f"Cannot fetch blob {blob_identifier} from blobs_in_frame of length {len(blobs_prior)} (#{frame_identifier} frame)")
            else:
                return blob


    def _annotate_output_of_parallel_computations_in_blobs(self, data):
        for i, process in enumerate(data):
            for entry in tqdm(process, desc=f"Annotating output of compute_overlapping_between_subsequent_frames_single_job {i}"):
                (frame_i, frame_i_index), (frame_ip1, frame_ip1_index) = entry
                blob_0 = self.fetch_blob(frame_i, frame_i_index)
                blob_1 = self.fetch_blob(frame_ip1, frame_ip1_index)
                blob_0.now_points_to(blob_1)

    def compute_overlapping_between_subsequent_frames_parallel(
        self, n_jobs=None
    ):

        if n_jobs is None:
            n_jobs = conf.NUMBER_OF_JOBS_FOR_CONNECTING_BLOBS

        self._annotate_location_of_blobs()
        FRAME_WITH_FIRST_BLOB=self._start_end_with_blobs[0]
        FRAME_WITH_LAST_BLOB=self._start_end_with_blobs[1]
        number_of_frames = len(self.blobs_in_video)


        starts = list(
            range(
                FRAME_WITH_FIRST_BLOB,
                FRAME_WITH_LAST_BLOB+1,
                conf.BLOB_CONNECTION_PROCESS_SIZE,
            )
        )
        ends = starts[1:] + [FRAME_WITH_LAST_BLOB+1]

        # Prepare list of arguments (args) for the pool of processes
        # Each element in this list is a tuple with the arguments
        # each worker will receive

        blobs = [
            self.blobs_in_video[starts[i] : ends[i]]
            for i in range(len(starts))
        ]
        starts_ends = [(starts[i], ends[i]) for i in range(len(starts))]

        function = [
            compute_overlapping_between_two_subsequent_frames,
        ] * len(starts)
    
        args = list(zip(blobs, function, starts_ends))

        with multiprocessing.Pool(n_jobs) as p:
            output = p.starmap(
                compute_overlapping_between_subsequent_frames_single_job, args
            )

        assert self.blobs_in_video[FRAME_WITH_FIRST_BLOB][0].frame_number == output[0][0][0][0]
        assert (
            self.blobs_in_video[FRAME_WITH_LAST_BLOB][0].frame_number
            == FRAME_WITH_LAST_BLOB
        )

        self._annotate_output_of_parallel_computations_in_blobs(output)
        self.stitch_parallel_blocks(starts, ends)
        assert number_of_frames == len(self.blobs_in_video)


    def stitch_parallel_blocks(self, starts, ends):
        """
        For every pair of consecutive frames in ends[i] and starts[i+1]
        compute the blob overlap
        This function is useful to connect blobs occurring in frames either
        at the start or the end of a block of frames processed in parallel
        The result is these blocks are "stitched" together
        """

        for i in tqdm(range(len(ends) - 1)):
            assert (ends[i]) == starts[i + 1]
            compute_overlapping_between_two_subsequent_frames(
                self.blobs_in_video[ends[i] - 1],
                self.blobs_in_video[starts[i + 1]],
            )