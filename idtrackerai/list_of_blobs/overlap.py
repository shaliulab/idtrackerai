import itertools
import numpy as np

def compute_overlapping_between_two_subsequent_frames(
    blobs_before, blobs_after, queue=None, do=True, threshold=None, **kwargs
):
    data = []

    for ((blob_0_i, blob_0), (blob_1_i, blob_1)) in itertools.product(
        enumerate(blobs_before), enumerate(blobs_after)
    ):
        overlaps = blob_0.overlaps_with(blob_1, **kwargs)
        # print(f"Overlap computed in {end_overlap - start_overlap} seconds")
        if overlaps:
            result = (
                (blob_0.frame_number, blob_0_i),
                (blob_1.frame_number, blob_1_i),
            )

            if do:
                blob_0.now_points_to(blob_1)

            elif queue is None:
                data.append(result)
            else:
                queue.put(result)

    return data


def compute_overlapping_between_two_subsequent_frames_with_ratio_threshold(
    blobs_before, blobs_after, queue=None, do=True, threshold=10, use_fragment_transfer_info=False
):
    if threshold is None:
        return compute_overlapping_between_two_subsequent_frames(
            blobs_before,
            blobs_after,
            queue=queue,
            do=do,
            use_fragment_transfer_info=use_fragment_transfer_info,
        )
    data = []

    if len(blobs_before) == 0 or len(blobs_after) == 0:
        return data

    for blob_0_i, blob_0 in enumerate(blobs_before):
        fractions_of_blob_0 = []
        for blob_1_i, blob_1 in enumerate(blobs_after):
            fractions_of_blob_0.append(blob_0.overlaps_with_fraction(blob_1))
            if use_fragment_transfer_info:
                if blob_0.fragment_transfer_overlaps_with(blob_1):
                    fractions_of_blob_0[-1] = 1.0
                
        
        best_blobs = np.argsort(fractions_of_blob_0)[::-1].tolist()
        
        blob_index = best_blobs[0]

        if fractions_of_blob_0[blob_index] > 0:
            result = (
                (blob_0.frame_number, blob_0_i),
                (blob_1.frame_number, blob_index),
            )
            
            if do:
                blob_1 = blobs_after[blob_index]
                blob_0.now_points_to(blob_1)
            elif queue is None:
                data.append(result)
            else:
                queue.put(result)

            previous_index = best_blobs[0]
            for blob_index in best_blobs[1:]:
                if (fractions_of_blob_0[blob_index] * threshold) > fractions_of_blob_0[previous_index]:
                    blob_1 = blobs_after[blob_index]
                    result = (
                        (blob_0.frame_number, blob_0_i),
                        (blob_1.frame_number, blob_index),
                    )
                    if do:
                        blob_0.now_points_to(blob_1)
                    elif queue is None:
                        data.append(result)
                    else:
                        queue.put(result)
                    previous_index = blob_index

                else:
                    break

    return data