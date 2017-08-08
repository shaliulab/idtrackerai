from get_data_nose_detector import GetTrainData
from train_nose_detector import TrainNoseDetector
from test_nose_detector import TestNoseDetector

video_path = '/home/lab/Desktop/TF_models/IdTrackerDeep/videos/test/1.avi'

# data = GetTrainData(video_path, save_flag = False)
# print('number of images ', len(data.data_dict['labels']))
# trained_model = TrainNoseDetector(video_path, num_epochs = 5, data_dict = data.data_dict, label_keyword = 'line_labels', plot_flag = True)
test = TestNoseDetector(video_path)
test.feed_images()
