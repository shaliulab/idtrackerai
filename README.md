# idtrackerai
## Tracking all individuals with correct identities in large animal collectives

idtracker.ai allows to track animals in small and large collectives using convolutional neural networks.

## Requirements

idtracker.ai has been tested under the following specifications:

- Operating system: 64bit linux Mint 18.1
- CPU: Core(TM) i7-7700K CPU @4.20GHz 6 core Intel(R) or Core(TM) i7-6800K CPU @3.40GHz 4 core
- GPU: Nvidia TITAN X or Nvidia GeForce GTX 1080 Ti
- RAM: 32Gb (for small groups) or 128Gb (for large groups)
- Disk: 1TB SSD

idtracker.ai is coded in python 2.7 and its core uses Tensorflow libraries
(version 1.2.0). Due to the intense use of deep neural networks, we recommend using
 a computer with a dedicated NVIDA GPU supporting compute capability 3.0 or higher. Note that the parts of the algorithm using Tensorflow libraries will run faster with a GPU. If a GPU
is not installed on the computer the CPU version of tensorflow will be installed
but the speed of the tracking will be highly affected.

## Installation.

The installation of idtracker.ai requires some amout of interaction with the linux 
terminal. Read the following paragraph only if your are not familiar using the terminal in 
linux operating systems.

In linux Mint you can open one using the icon ">_" in the bottom panel. We provide the 
commands needed to install idtracker.ai from the terminal. You can type them directly 
in the command line and press ENTER to execute them. Use the right button in your 
mouse to copy and paste commands from the instructinos to the terminal. (NOTE: 
do not use the shortcut Ctrl+C and Ctrl+V as they do not work in the terminal)

### Pre-installation checks

To install the GPU version of idtracker.ai make sure that NVIDIA drivers,
CUDA Toolkit 8.0 and the corresponding CuDNN v5.1 libraries for CUDA 8.0 are
installed in your computer.

To check that your computer detects the GPU and the drivers are correctly installed
type

    nvidia-smi

in your terminal. You should get an output similar to this:

    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 384.111                Driver Version: 384.111                   |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  GeForce GTX 108...  Off  | 00000000:03:00.0  On |                  N/A |
    |  0%   35C    P5    24W / 250W |    569MiB / 11171MiB |      1%      Default |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |    0      1266      G   /usr/lib/xorg/Xorg                           280MiB |
    |    0      1712      G   cinnamon                                     119MiB |
    |    0      2402      G   ...vendor-id=0x10de --gpu-device-id=0x1b06    50MiB |
    |    0      3869      G   ...-token=B623FBCDF5B2B7F30ECB74EC9ADEFC8F   117MiB |
    +-----------------------------------------------------------------------------+

To check that CUDA Toolkit 8.0 is properly installed type

    nvcc -V

in your terminal. You should get an output similar to this:

    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2016 NVIDIA Corporation
    Built on Tue_Jan_10_13:22:03_CST_2017
    Cuda compilation tools, release 8.0, V8.0.61

To check that the correct version of CuDNN is installed type

    cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2

in your terminal. You should get an output similar to this:

    #define CUDNN_MAJOR      5
    #define CUDNN_MINOR      1
    #define CUDNN_PATCHLEVEL 10
    --
    #define CUDNN_VERSION    (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)

    #include "driver_types.h"

For further information please check the NVIDIA requirements to run TensorFlow with GPU support
for Tensorflow 1.2.0 [here](https://www.tensorflow.org/versions/r1.2/install/install_linux).


### Installation

Using the terminal create a folder named idtrackerai in your Home directory by typing 
    
    mkdir idtrackerai
    
Move to that folder by typing 

    cd idtrackerai

Download the file [install.sh](https://gitlab.com/polavieja_lab/idtrackerai/raw/write_setup/install.sh) 
using the following command.

    wget https://gitlab.com/polavieja_lab/idtrackerai/raw/write_setup/install.sh

Execute the install file:

    ./install.sh

If install.sh does not have executable permission type

    chmod u+x install.sh
    ./install.sh

install.sh will first install [miniconda](https://conda.io/miniconda.html),
you will be asked review the license agreement and accept the terms and conditions.  
Review the license agreement by pressing ENTER and accept the terms typing "yes".

Then you will be asked to confirm the location of the installation. Press ENTER
to continue with the default installation.

If a version of miniconda is not yet installed you will be asked to prepend the install
location to PATH in your .bashrc file. Type "yes" to continue with the default installation.
Otherwise you will get an ERROR indicating that the directory already exists, but the
installation of idtrackerai will continue.

At the end of the installation activate the environment and launch the GUI:

    source activate idtrackerai-environment
    idtrackeraiGUI
    
Go to the [Quick start](http://idtracker.ai/quickstart.html) and follow the instructions 
to track a simple example video. 

If the installation did not succeed try proceeding step by step, by running
the following commands in your terminal:

    wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
    chmod u+x Miniconda2-latest-Linux-x86_64.sh
    ./Miniconda2-latest-Linux-x86_64.sh
    source ~/.bashrc
    wget https://gitlab.com/polavieja_lab/idtrackerai/raw/write_setup/environment-mint18.1.yml
    conda env create -f environment-mint18.1.yml
    source activate idtrackerai-environment
    git clone https://gitlab.com/polavieja_lab/idtrackerai.git
    pip install idtrackerai/.
    source activate idtrackerai-environment
    garden install matplotlib
    chmod u+x idtrackerai/run.sh

### Uninstall and remove software





## Documentation and examples of tracked videos
http://idtracker.ai

## Contributors
* Mattia G. Bergomi
* Francisco Romero-Ferrero
* Francisco J.H. Heras

## License
This is idtracker.ai a multiple animals tracking system
described in [1].
Copyright (C) 2017- Bergomi, M.G., Romero-Ferrero, F., Heras, F.J.H.

idtracker.ai is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details. In addition, we require
derivatives or applications to acknowledge the authors by citing [1].

A copy of the GNU General Public License is available [here](LICENSE).

For more information please send an email (idtrackerai@gmail.com) or
use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.

**[1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., De Polavieja, G.G.,
(2018). idtracker.ai: Tracking all individuals with correct identities in large
animal collectives (submitted)**
