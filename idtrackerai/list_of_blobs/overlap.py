import itertools

def compute_overlapping_between_two_subsequent_frames(
    blobs_before, blobs_after, queue=None, do=True
):
    data = []

    for ((blob_0_i, blob_0), (blob_1_i, blob_1)) in itertools.product(
        enumerate(blobs_before), enumerate(blobs_after)
    ):
        overlaps = blob_0.overlaps_with(blob_1)
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