# idTrackerai

Python 2.7

### Installation in linux Mint 18.1

* cd
* sudo apt install python-pip
* sudo apt-get install python-dev
* sudo apt-get install python-setuptools
* (install opencv 2.4) curl -s "https://raw.githubusercontent.com/arthurbeggs/scripts/master/install_apps/install_opencv2.sh" | bash
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

### Requirements (see setup.py):

* Cython >= 0.26.1
* pygame >= 1.9.3
* numpy >= 1.13.0
* natsort >= 5.0.2
* matplotlib >= 2.1
* seaborn >= 0.8
* tqdm >= 4.19
* joblib >= 0.11
* scikit-learn >= 0.19
* PyAutoGUI >= 0.9.36
* pyyaml >= 3.12
* psutil >= 5.4.3
* h5py >= 2.7.0
* Kivy >= 1.10.0
* Kivy-Garden >= 0.1.4
* matplotlib >= 2.0.0
* msgpack-numpy >= 0.3.9
* msgpack-python >= 0.4.8
* natsort >= 5.0.2
* pandas >= 0.20.2
* PyYAML >= 3.12
* scipy >= 0.19.0
* sklearn >= 0.0
* protobuf >= 3.4.0
* tables >= 3.3.0
* dask >= 0.17.0
* tensorflow-gpu >= 1.4
