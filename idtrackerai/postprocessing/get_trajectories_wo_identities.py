from tqdm import tqdm
import numpy as np
import os

def track_wo_identities(video, list_of_blobs):
    trajectories = np.ones((video.number_of_frames, video.number_of_animals, 2))*np.nan
    blobs_in_video = list_of_blobs.blobs_in_video
    identifiers_prev = np.arange(video.number_of_animals).astype(np.float32)
    for frame_number, blobs_in_frame in enumerate(tqdm(blobs_in_video[:-1], "creating trajectories")):
        identifiers_next = [b.fragment_identifier for b in blobs_in_video[frame_number+1]]
        for blob_number, blob in enumerate(blobs_in_frame):
            if blob.is_an_individual:
                if blob.fragment_identifier in identifiers_prev:
                    column = np.where(identifiers_prev == blob.fragment_identifier)[0][0]
                else:
                    column = np.where(np.isnan(identifiers_prev))[0][0]
                    identifiers_prev[column] = blob.fragment_identifier

                trajectories[frame_number, column, :] = blob.centroid

                if blob.fragment_identifier not in identifiers_next:
                    identifiers_prev[column] = np.nan
                blob._user_generated_identity = int(column)

    video.create_trajectories_folder()
    trajectories_dict = {'trajectories': trajectories,
                         'id_probabilities': None,
                         'git_commit': video.git_commit,
                         'video_path': video.video_path,
                         'frames_per_second': video.frames_per_second,
                         'body_length': video.median_body_length}
    np.save(os.path.join(video.trajectories_folder, 'trajectories_no_identities.npy'), trajectories_dict)
