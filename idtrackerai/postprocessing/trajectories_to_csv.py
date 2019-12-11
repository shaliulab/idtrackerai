import os
import sys
import numpy as np
import json

def save_array_to_csv(path, array, key=''):
    array = np.squeeze(array)
    if array.ndim == 3:
        array_reshaped = np.reshape(array, (-1, array.shape[1] * array.shape[2]))
        array_header = ','.join([coord + str(i) for i in range(1, array.shape[1] + 1) for coord in ['x', 'y']])
        np.savetxt(path, array_reshaped, delimiter=',', header=array_header)

    elif array.ndim == 2:
        array_header = ','.join([key + str(i) for i in range(1, array.shape[1] + 1)])
        np.savetxt(path, array, delimiter=',', header=array_header)


def convert_trajectories_file_to_csv_and_json(trajectories_file):
    trajectories_dict = np.load(trajectories_file, allow_pickle=True).item()

    file_name = os.path.splitext(trajectories_file)[0]

    if trajectories_dict is None:
        trajectories_dict = np.load(trajectories_file, allow_pickle=True).item()

    attributes_dict = {}
    for key in trajectories_dict:
        if isinstance(trajectories_dict[key], np.ndarray):
            key_csv_file = file_name + '.' + key + '.csv'
            save_array_to_csv(key_csv_file, trajectories_dict[key], key=key)

        else:
            attributes_dict[key] = trajectories_dict[key]

    attributes_file_name = file_name + '.attributes.json'
    with open(attributes_file_name, 'w') as fp:
        json.dump(attributes_dict, fp)


if __name__ == '__main__':
    path = sys.argv[1]

    for root, folders, files in os.walk(path):
        for file in files:
            if 'trajectories' in file and '.npy' in file:
                trajectories_file = os.path.join(root, file)
                print("Converting {} to .csv and .json".format(trajectories_file))
                convert_trajectories_file_to_csv_and_json(trajectories_file)






