import os
import idtrackerai.constants as cons

DATA_FOLDER = os.path.join(os.path.dirname(cons.IDTRACKERAI_FOLDER), "data")
TEST_VIDEO_COMPRESSED_PATH = os.path.join(
    DATA_FOLDER,
    "example_video_compressed",
    "idtrackerai_video_example_compressed.avi",
)


def test_data_exists():
    assert os.path.isfile(TEST_VIDEO_COMPRESSED_PATH)
