# idTrackerDeep

Python 2.7

### Installation in linux Mint 18.1

* cd
* sudo apt install python-pip
* sudo apt-get install python-dev
* sudo apt-get install python-setuptools
* sudo pip install virtualenv (http://python-guide-pt-br.readthedocs.io/en/latest/dev/virtualenvs/)
* sudo pip install virtualenvwrapper (http://python-guide-pt-br.readthedocs.io/en/latest/dev/virtualenvs/)
* follow http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation to install CUDA® Toolkit 8.0
* check NVIDIA drivers compatible with cuda toolkit 8.0
* download cuDNN v5.1 libraries and copy them in to the correspoding cuda-8.0 folders (typically usr/local/cuda-8.0/include and usr/local/cuda-8.0/lib64)
* mkvirtualenv --system-site-packages idTracker100 
* workon idTracker100
* sudo apt-get install python-opencv (version 2.4.9.1)
* sudo apt-get install git
* git clone http://paco@git.yalp.io/mattia/IdTrackerDeep.git (or whatever HTTP link)
* sudo apt-get install python-tk
* sudo pip install requirements.txt

### Dependencies needed for the project:

* os
* glob
* time
* multiprocessing
* itertools
* tensorflow > 1
* cPickle
* math
* scipy
* re
* pprint
* numpy
* matplotlib
* cv2 > 2.4.8
* pandas
* joblib
* scikit-image
* cython
* natsort
* tkinter
* msgpack-numpy
* msgpack
* h5py