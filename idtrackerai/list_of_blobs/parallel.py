import time
import multiprocessing
import logging
import math
from tqdm import tqdm
from confapp import conf

from .overlap import compute_overlapping_between_two_subsequent_frames

logger = logging.getLogger("__main__.list_of_blobs.parallel")

n_jobs=conf.NUMBER_OF_JOBS_FOR_CONNECTING_BLOBS

def compute_overlapping_between_subsequent_frames_single_job(
    blobs_in_video, f, queue, start_and_end=None
):

    """
    Arguments:

        start (tuple): Index of the original blobs_in_video from which this job will process
        Only used in the progress bar, so not essential
    """

    if start_and_end is None:
        desc="Connecting blobs "
    else:
        desc=f"Connecting blobs from {start_and_end[0]} to {start_and_end[1]} "

    for frame_i in tqdm(
        range(1, len(blobs_in_video)), desc=desc
    ):
        f(blobs_in_video[frame_i - 1], blobs_in_video[frame_i], queue)


class ParallelBlobOverlap:
    """
    A mixin class that implements the parallel processing of blobs in a video
    making full use of the computer's CPU (instead of just 1) 
    """

    @staticmethod
    def make_process_name(start, end):
        return f"Process [{start} - {end})"

    def generate_parallel_processes(self, starts, ends, queue, max_jobs):

        for i in range(len(starts)):
            process_name = self.make_process_name(starts[i], ends[i])
            process = multiprocessing.Process(
                target=compute_overlapping_between_subsequent_frames_single_job,
                name=process_name,
                args=(
                    self.blobs_in_video[starts[i] : ends[i]],
                    compute_overlapping_between_two_subsequent_frames,
                    queue,
                    (starts[i], ends[i])
                ),
            )

            self._processes[process_name] = process

            if i >= max_jobs:
                break
        
        return i

    def partition_blobs_across_processes(self):
        """
        Given a total number of frames and a number of frames to be processed on each parallel process,
        this function computes the start and end of the frame intervals that should be used
        to split the dataset and process it in parallel
        so:
        * starts[i] is the first frame to be processed on the ith parallel process
        * ends[i] is the frame following the last frame to be to be processed on the ith parallel process
        following Python's convention of declaring closed-open intervals. i.e. [start, end)
        """

        #math.ceil(self.number_of_frames / n_jobs)

        starts = list(range(0, self.number_of_frames, conf.BLOB_CONNECTION_PROCESS_SIZE))
        ends = [starts[i] + conf.BLOB_CONNECTION_PROCESS_SIZE for i in range(len(starts))]
        return starts, ends


    def compute_overlapping_between_subsequent_frames_parallel(self):

        self._processes = {}
        logger.info(
            f"""
            Computing overlap between subsequent frames in parallel using {conf.NUMBER_OF_JOBS_FOR_CONNECTING_BLOBS} jobs
            """
        )
        starts, ends = self.partition_blobs_across_processes()
        queue = multiprocessing.Queue(maxsize=0)
        n_started_jobs = self.generate_parallel_processes(starts, ends, queue, conf.NUMBER_OF_JOBS_FOR_CONNECTING_BLOBS)


        # compute the blob overlap in parallel
        for process_name, process in self._processes.items():
            # this call starts a process / worker in the background
            # so this whole for loop takes ms to run
            process.start()
            # Here we need to figure out when n_jobs processes have
            # been started and dont start more until one of them is done
            # Ideally the queue and the remaining processes start can be managed
            # in the same function

        # receive the results of the parallel processes!!
        n_started_jobs=self.process_blob_overlap_queue(queue, starts, ends, n_started_jobs)

        if queue.qsize() != 0:
            logger.warning(
                "Not all blobs have been annotated. idtrackerai may hang"
            )

        for process_name, process in self._processes.items():
            # if the queue still has stuff in it, the program
            # hangs here
            process.join()

        if len(starts) != 1:
            self.stitch_parallel_blocks(starts, ends)

        # remove processes so the list_of_blobs can be pickled!
        processes = list(self._processes.keys())
        for p in processes:
            del self._processes[p]


    @property
    def number_of_running_jobs(self):

        accum = 0
        for process_name, process in self._processes.items():
            # TODO:
            # Is this the way to verify a process is running? (i.e. taking CPU power)
            if process.is_alive():
                accum += 1

        return accum


    def process_blob_overlap_queue(self, queue, starts, ends, n_started_jobs):
            """
            Process the queue in the main thread
            This queue is filled by all the parallel process
            which are computing which blob is next to previous to any given blob
            However, the annotation is only applied here,
            because that step can only
            be done in the main thread.
            Fortunately, the time consuming and CPU intensive step is the computation
            (encapsulated in overlaps_with)
            and not the annotation (encapsulated in now_points_to)
            so it's fine to run the latter in only one thread (the main)
            """

            while queue.qsize() != 0:
                (
                    (frame_i, frame_i_index),
                    (frame_ip1, frame_ip1_index),
                ) = queue.get()
                blob_0 = self.blobs_in_video[frame_i][frame_i_index]
                blob_1 = self.blobs_in_video[frame_ip1][frame_ip1_index]
                blob_0.now_points_to(blob_1)
                logger.debug(f"{blob_0.frame_number}:{blob_0.in_frame_index} -> {blob_1.frame_number}:{blob_1.in_frame_index}")


                while self.number_of_running_jobs < conf.NUMBER_OF_JOBS_FOR_CONNECTING_BLOBS and \
                    len(self._processes) < len(starts):

                    self.generate_parallel_processes([starts[n_started_jobs]],[ends[n_started_jobs]], queue, 1)
                    self._processes[self.make_process_name(starts[n_started_jobs], ends[n_started_jobs])].start()
                    n_started_jobs+=1
               
                if queue.qsize() == 0:
                    time.sleep(2)
                    # make extra sure the queue
                    # has had time to get anything new
                    # before exiting this function
                    # once, we exit, anything new will stay there
                    # and the program will hang

            return n_started_jobs


    def stitch_parallel_blocks(self, starts, ends):
        """
        For every pair of consecutive frames in ends[i] and starts[i+1]
        compute the blob overlap
        This function is useful to connect blobs occurring in frames either
        at the start or the end of a block of frames processed in parallel
        The result is these blocks are "stitched" together
        """

        for i in tqdm(range(len(ends) - 1)):
            assert (ends[i] - 1 + 1) == starts[i + 1]
            compute_overlapping_between_two_subsequent_frames(
                self.blobs_in_video[ends[i] - 1],
                self.blobs_in_video[starts[i + 1]],
            )