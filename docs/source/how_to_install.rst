Installation and requirements
=============================

^^^^^^^^^^^^
Requirements
^^^^^^^^^^^^

The system has been developed and tested under the following specifications:

- Operating system: 64bit linux Mint 18.1
- CPU: Core(TM) i7-7700K CPU @4.20GHz   6 core Intel(R) Core(TM) i7-6800K CPU @3.40GHz
- GPU: Nvidia TITAN X or Nvidia GeForce GTX 1080 Ti
- RAM: 32Gb (for small groups) or 128Gb (for large groups)
- Disk: 1TB SSD

idtracker.ai is coded in python 2.7 and its core uses OpenCV 2 python wrappers
(version 2.13.4) and Tensorflow libraries (version 1.2.0). Due to the intense
use of deep neural networks, we recommend using a computer with a dedicated
NVIDA GPU supporting compute capability 3.0 or higher. Note that the parts of
the algorithm using Tensorflow libraries will run faster with a GPU. If a GPU
is not installed on the computer the CPU version of tensorflow will be installed
but the speed of the tracking would be highly affected.

^^^^^^^^^^^^
Installation
^^^^^^^^^^^^

*This installation has been tested under linux Mint 18.1.*

Download :download:`install.sh <../../install.sh>` file and create a folder idtrackerai.

    cd idtrackerai
    ./install.sh

If install.sh does not have executable permission type

    chmod u+x install.sh
    ./install.sh

install.sh will first install miniconda, you will be asked to accept the terms
and conditions, then will download this repository and install the package.

At the end of the installation activate the environment and launch the GUI:

    conda activate idtrackera-environment
    idtrackeraiGUI

If the installation did not succeed try proceeding step by step, by running
the following commands in your terminal:

    wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
    chmod +x Miniconda2-latest-Linux-x86_64.sh
    ./Miniconda2-latest-Linux-x86_64.sh
    source ~/.bashrc
    wget http://git.yalp.io/mattia/IdTrackerDeep/raw/write_setup/environment-mint18.1.yml
    conda env create -f environment-mint18.1.yml
    source activate idtrackerai-environment
    git clone http://git.yalp.io/mattia/IdTrackerDeep.git
    pip install IdTrackerDeep/.
    chmod +x IdTrackerDeep/run.sh
