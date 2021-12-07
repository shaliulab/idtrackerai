import argparse
import datetime
import os.path

from confapp import conf

from idtrackerai.list_of_blobs import ListOfBlobs


def pick_blobs_collection(session_folder):
    
    blobs_collection = os.path.join(session_folder, "preprocessing", "blobs_collection_no_gaps.npy")
    if os.path.exists(blobs_collection):
        return blobs_collection
    else:
        blobs_collection = os.path.join(session_folder, "preprocessing", "blobs_collection.npy")
        if os.path.exists(blobs_collection):
            return blobs_collection
        else:
            raise ValueError(f"No blobs collection found for {session_folder}")


def get_parser(ap=None):

    if ap is None:
        ap = argparse.ArgumentParser()

    ap.add_argument("session_folder", required=True, type=str)
    ap.add_argument("--n_jobs", dest="n_jobs", required=False, type=int, default=None)
    return ap


def main(ap=None, args=None):

    if args is None:
        if ap is None:
            ap = get_parser()
        
        args = ap.parse_args()
    
    blobs_collection = pick_blobs_collection(args.session_folder)
    timestamp_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    blobs_collection_dest = os.path.join(
        os.path.dirname(blobs_collection),
        timestamp_now + "_" + os.path.basename(blobs_collection)
    )
    list_of_blobs = ListOfBlobs.load(blobs_collection)

    if args.n_jobs is None:
        n_jobs = getattr(conf, "NUMBER_OF_JOBS_FOR_CONNECTING_BLOBS", False)
    else:
        n_jobs = args.n_jobs

    if n_jobs is False:
        list_of_blobs.compute_overlapping_between_subsequent_frames()
    else:
        list_of_blobs.compute_overlapping_between_subsequent_frames(n_jobs)
    
    
    return list_of_blobs.save(blobs_collection_dest)


def reconnect(*args, **kwargs):
    return main(*args, **kwargs)