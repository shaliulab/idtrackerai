from __future__ import absolute_import, print_function, division
import numpy as np
import cv2

if __name__ == "__main__":
    video_path = ''
    video_object = np.load(video_path).item()
    trajectories  = np.load(file).item()['trajectories']
    cap = cv2.VideoCapture(video_object.video_path)
    numFrames = video_object._num_frames

    global currentSegment, cap
    currentSegment = 0
    cv2.namedWindow('frame_by_frame_identity_inspector')
    defFrame = 0
    colors = get_spaced_colors_util(video_object.number_of_animals)


    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    name = video_object._session_folder +'/tracked.avi'
    out = cv2.VideoWriter(name, fourcc, 32.0, (video_object._width, video_object._height))

    def scroll(trackbarValue):
        global frame, currentSegment, cap
        sNumber = video_object.in_which_episode(trackbarValue)
        sFrame = trackbarValue
        if sNumber != currentSegment: # we are changing segment
            print('Changing segment...')
            currentSegment = sNumber
            if video_object._paths_to_video_segments:
                cap = cv2.VideoCapture(video_object._paths_to_video_segments[sNumber])

        #Get frame from video file
        if video_object._paths_to_video_segments:
            start = video_object._episodes_start_end[sNumber][0]
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,sFrame - start)
        else:
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbarValue)
        ret, frame = cap.read()
        frameCopy = frame.copy()

        blobs_in_frame = blobs_in_video[trackbarValue]
        for b, blob in enumerate(blobs_in_frame):

            blobs_pixels = get_n_previous_blobs_attribute(blob,'pixels',number_of_previous)[::-1]
            blobs_identities = get_n_previous_blobs_attribute(blob,'_identity',number_of_previous)[::-1]
            for i, (blob_pixels, blob_identity) in enumerate(zip(blobs_pixels,blobs_identities)):
                pxs = np.unravel_index(blob_pixels,(video_object._height,video_object._width))
                if i < number_of_previous-1:
                    frame[pxs[0],pxs[1],:] = np.multiply(colors[blob_identity],.3).astype('uint8')+np.multiply(frame[pxs[0],pxs[1],:],.7).astype('uint8')
                else:
                    frame[pxs[0],pxs[1],:] = frameCopy[pxs[0],pxs[1],:]

            #draw the centroid
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.circle(frame, tuple(blob.centroid), 2, colors[blob._identity], -1)
            if blob._assigned_during_accumulation:
                # we draw a circle in the centroid if the blob has been assigned during accumulation
                cv2.putText(frame, str(blob._identity),tuple(blob.centroid), font, 1,colors[blob._identity], 3)
            elif not blob._assigned_during_accumulation:
                # we draw a cross in the centroid if the blob has been assigned during assignation
                # cv2.putText(frame, 'x',tuple(blob.centroid), font, 1,colors[blob._identity], 1)
                cv2.putText(frame, str(blob._identity),tuple(blob.centroid), font, .5,colors[blob._identity], 3)

            if not save_video:
                print("\nblob ", b)
                print("identity: ", blob._identity)
                print("assigned during accumulation: ", blob.assigned_during_accumulation)
                if not blob.assigned_during_accumulation and blob.is_a_fish_in_a_fragment:
                    try:
                        print("frequencies in fragment: ", blob.frequencies_in_fragment)
                    except:
                        print("this blob does not have frequencies in fragment")
                print("P1_vector: ", blob.P1_vector)
                print("P2_vector: ", blob.P2_vector)


        if not save_video:
            # frame = cv2.resize(frame,None, fx = np.true_divide(1,4), fy = np.true_divide(1,4))
            cv2.imshow('frame_by_frame_identity_inspector', frame)
            pass
        else:
            out.write(frame)

    cv2.createTrackbar('start', 'frame_by_frame_identity_inspector', 0, numFrames-1, scroll )

    scroll(1)
    cv2.setTrackbarPos('start', 'frame_by_frame_identity_inspector', defFrame)
    # cv2.waitKey(0)
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)

    if save_video:
        for i in tqdm(range(video_object._num_frames)):
            scroll(i)

    cv2.waitKey(0)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    save_video = getInput('Saver' , 'Do you want to save a copy of the tracked video? [y]/n')
    if not save_video or save_video == 'y':
        frame_by_frame_identity_inspector(video, blobs_in_video, save_video = True)
    else:
        return
