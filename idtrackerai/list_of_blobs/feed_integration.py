def overlap_with_a_distant_blob(blob, list_of_blobs, dest_fn, move, step, number_of_animals, do=True):
    assert step != 0
    i = list_of_blobs.blobs_in_video[dest_fn].index(blob)
    while True:
        distant_frame = dest_fn+move
        if distant_frame >= len(list_of_blobs.blobs_in_video):
            break
        distant_blobs_in_frame =  list_of_blobs.blobs_in_video[distant_frame]
        if len(distant_blobs_in_frame) != number_of_animals:
            move=move+step
        else:
            for j, distant_blob in enumerate(distant_blobs_in_frame):
                if blob.overlaps_with(distant_blob, use_fragment_transfer_info=True, only_use_fragment_transfer_info=True):
                    if move > 0:
                        fragment_joined=False
                        if blob.next:
                            for next_blob in blob.next:
                                if getattr(blob, "source_fragment_identifier", None) == \
                                    getattr(next_blob, "source_fragment_identifier", None):
                                        fragment_joined=True
                                        
                        if not fragment_joined:
                            if do:
                                blob.reset_next()
                                distant_blob.reset_previous()
                                blob.now_points_to(distant_blob)
                                blob.bridge_start = (distant_blob.frame_number, j)
                                distant_blob.bridge_end = (blob.frame_number, i)
                            else:
                                print(f"{blob} -> {distant_blob}")

                    if move < 0:
                        fragment_joined=False
                        if blob.previous:
                            for previous_blob in blob.previous:
                                if getattr(blob, "source_fragment_identifier", None) == \
                                    getattr(previous_blob, "source_fragment_identifier", None):
                                        fragment_joined=True
                                        
                        if not fragment_joined:
                            if do:
                                distant_blob.reset_next()
                                blob.reset_previous()
                                distant_blob.now_points_to(blob)
                                blob.bridge_end = (distant_blob.frame_number, j)
                                distant_blob.bridge_start = (blob.frame_number, i)
                            else:
                                print(f"{distant_blob} -> {blob}")

            # we stop at the first frame where all animals are observed again!
            return


def bypass_crossing(list_of_blobs, blobs_in_frame, next_blobs_in_frame, frame_number, number_of_animals, do=True):
    blobs_where_next_is_nothing=[]
    blobs_where_previous_is_nothing=[]
    

    if len(blobs_in_frame) > 0 and len(next_blobs_in_frame) > 0:
        if len(blobs_in_frame) != number_of_animals and len(next_blobs_in_frame) != number_of_animals:
            return

        # there is a fragment break in the next frame
        elif len(next_blobs_in_frame) != number_of_animals:
            for blob in blobs_in_frame:
                if not blob.next or len(blob.next[0].previous) != 1:
                    blobs_where_next_is_nothing.append(blob)

            for blob in next_blobs_in_frame:
                if not blob.previous or len(blob.previous) != 1:
                    blobs_where_previous_is_nothing.append(blob)

            for blob in blobs_where_next_is_nothing:
                blob.reset_next()
                overlap_with_a_distant_blob(
                    blob, list_of_blobs, frame_number,
                    move=+2, step=+1,
                    number_of_animals=number_of_animals,
                    do=do
                )

            for blob in blobs_where_previous_is_nothing:
                blob.reset_previous()


        # there was a fragment break in the previous frame
        elif len(blobs_in_frame) != number_of_animals:
            for blob in blobs_in_frame:
                if not blob.next or len(blob.next) != 1:
                    blobs_where_next_is_nothing.append(blob)

            for blob in next_blobs_in_frame:
                if not blob.previous or len(blob.previous[0].next) != 1:
                    blobs_where_previous_is_nothing.append(blob)

            for blob in blobs_where_next_is_nothing:
                blob.reset_next()
                
            for blob in blobs_where_previous_is_nothing:
                blob.reset_previous()
                overlap_with_a_distant_blob(
                    blob, list_of_blobs,
                    frame_number+1, move=-2, step=-1,
                    number_of_animals=number_of_animals,
                    do=do
                )

def bypass_crossings(list_of_blobs, number_of_animals, debug_fn=None):

    for dest_fn, blobs_in_frame in enumerate(list_of_blobs.blobs_in_video[:-1]):
        next_blobs_in_frame=list_of_blobs.blobs_in_video[dest_fn+1]
        if debug_fn is not None and dest_fn == debug_fn:
            import ipdb; ipdb.set_trace()
        bypass_crossing(list_of_blobs, blobs_in_frame, next_blobs_in_frame, frame_number=dest_fn, number_of_animals=number_of_animals)

