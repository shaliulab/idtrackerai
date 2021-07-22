import os
import idtrackerai.constants as cons


def test_data_exists():
    assert os.path.isfile(cons.COMPRESSED_VIDEO_PATH)
    assert os.path.isfile(cons.COMPRESSED_VIDEO_PATH_2)
