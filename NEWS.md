
idtrackerai/tracker/tracker.py
If conf.DISABLE_PROTOCOL_3 is True, the PROTOCOL_3 is skipped

idtrackerai/crossings_detection/dataset/crossings_dataset.py
Benchmark the function to check if there are crossings in the future or past

4.0.9
===============

Rerunning tracking on a session is possible now, i.e. idtrackerai supports overwriting the identities in the h5df file

The first frame of the first global fragment is assigned now the correct frame number (instead of 0) when the input dataset is an imgstore
