from __future__ import absolute_import, division, print_function
import os
import glob

if __name__ == '__main__':

    lib_directory = '/media/chronos/idZebLib_TU20170131/TU20170131/31dpf'
    lib_sub_dirs = glob.glob(lib_directory + '/*/')
    for lib_sub_dir in lib_sub_dirs:
        if os.path.isdir(lib_sub_dir + '/preprocessing'):
            if os.path.isfile(lib_sub_dir + '/preprocessing/portraits.pkl'):
                os.rename(lib_sub_dir + '/preprocessing/portraits.pkl',lib_sub_dir + '/preprocessing/portraits_cropped.pkl')
