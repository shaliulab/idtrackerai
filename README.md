# idtracker.ai

[idtracker.ai in arXiv](https://arxiv.org/abs/1803.04351)

## Hardware requirements

idtracker.ai has been tested in computers with the following specifications:

- Operating system: 64bit GNU/linux Mint 18.1
- CPU: Core(TM) i7-7700K CPU @4.20GHz 6 core Intel(R) or Core(TM) i7-6800K CPU @3.40GHz 4 core
- GPU: Nvidia TITAN X or GeForce GTX 1080 Ti
- RAM: 32Gb (for small groups) or 128Gb (for large groups)
- Disk: 1TB SSD

idtracker.ai is coded in python 2.7 and uses Tensorflow libraries
(version 1.2.0). Due to the intense use of deep neural networks, we recommend using
 a computer with a dedicated NVIDA GPU supporting compute capability 3.0 or higher.
 Note that the parts of the algorithm using Tensorflow libraries will run faster with a GPU. If a GPU
is not installed on the computer the CPU version of Tensorflow will be installed
but the speed of the tracking will be highly affected.

## Installation.

The installation of idtracker.ai requires some amount of interaction with the linux
terminal. Read the following paragraph only if your are not familiar with the terminal in
linux operating systems. 

In linux Mint you can open a terminal using the icon with the gray symbol ">_" on the left in the bottom bar.
We provide the commands needed to install idtracker.ai from the terminal.
In this documentation inputs to the terminal and outputs are shown inside of a box.
You can type them directly in the command line and press ENTER to execute them.
Right-click with your mouse to copy and paste commands from the instructions to the terminal.
(NOTE: do not use the shortcut Ctrl+C and Ctrl+V as they do not work in the terminal)

The time needed to install the system varies with the output of the pre-installation checks and the 
download speed of the network when cloning the repository and dowloading the dependencies. 
In our computers and network, the total installation time is typicall of 15 minutes. 

### Pre-installation checks

###### GPU Drivers, CUDA Toolkit and CuDNN library.

To install the GPU version of idtracker.ai first make sure that NVIDIA drivers,
CUDA Toolkit 8.0 and the corresponding CuDNN v5.1 libraries for CUDA 8.0, are
installed in your computer.

If you are using an NVIDIA GPU you can check that the drivers are properly
installed typing

    nvidia-smi

in your terminal. You should get an output that looks like:

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

It is important that the release of the Cuda compilation tools is 8.0.

To check that the correct version of CuDNN is installed type

    cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2

in your terminal. You should get an output similar to this:

    #define CUDNN_MAJOR      5
    #define CUDNN_MINOR      1
    #define CUDNN_PATCHLEVEL 10
    --
    #define CUDNN_VERSION    (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)

    #include "driver_types.h"

It is important that the CUDNN_MAJOR is 5 and the CUDNN_MINOR is 1

For further information please check the NVIDIA requirements to run TensorFlow with GPU support
for Tensorflow 1.2.0 [here](https://www.tensorflow.org/versions/r1.2/install/install_linux).

###### Python checks
**[NEW!!]** Make sure python-dev and python-pip are installed. You can install them running.

        sudo apt-get install python-pip python-dev 
        
###### Extra libraries needed for OpenCV

In the last installations we are experiencing problems when some libraries related to *ffmpeg*
are not installed. Please install them running.

        sudo apt install libavcodec-ffmpeg56
        sudo apt install libavformat-ffmpeg56 
        sudo apt install libswscale-ffmpeg3
  
###### Miniconda package manager

The installation process requires [miniconda](https://conda.io/miniconda.html) to be installed in your computer. Skip the next paragraphs if Miniconda2 or Miniconda3 are already installed.

To check whether miniconda is installed in your computer type

    conda

if you get the following output

    conda: command not found

miniconda is not installed in your computer. Follow the next instructions to install it.

Using the terminal, download the miniconda installation file

    wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh

and give it executable permissions.

    chmod u+x Miniconda2-latest-Linux-x86_64.sh

Execute the file

    ./Miniconda2-latest-Linux-x86_64.sh

You will be asked to review the license agreement and accept the terms and conditions.  
Review the license agreement by pressing ENTER and accept the terms typing "yes".

Then you will be asked to confirm the location of the installation. Press ENTER
to continue with the default installation. Finally you will be asked to prepend
the install location to PATH in your .bashrc file.
Type "yes" to continue with the default installation.

At the end of the installation close the terminal and open a new one.

### Installation

Using the terminal, download the file [install.sh](https://gitlab.com/polavieja_lab/idtrackerai/raw/1.0.0-alpha/install.sh)
using the following command.

    wget https://gitlab.com/polavieja_lab/idtrackerai/raw/1.0.0-alpha/install.sh

Give install.sh executable permissions by typing

    chmod u+x install.sh

and execute it

    ./install.sh

The installation can take several minutes. At the end of the process the last lines
of the terminal should show a message similar to this:

    Download done (21059 downloaded)
    Extracting...
    Installing new version...
    Done! garden.matplotlib is installed at: /home/rhea/.kivy/garden/garden.matplotlib
    Cleaning...

If the installation did not succeed try proceeding step by step, by running
the following commands in your terminal:

    wget https://gitlab.com/polavieja_lab/idtrackerai/raw/1.0.0-alpha/environment-mint18.1.yml
    conda env create -f environment-mint18.1.yml
    source activate idtrackerai-environment
    git clone https://gitlab.com/polavieja_lab/idtrackerai.git
    pip install idtrackerai/.
    source activate idtrackerai-environment
    garden install matplotlib
    
### Troubleshooting installation

**[NEW!!]** 
In some installations the libdc1394 is missing and OpenCV does not work. 
Install this library inside of the conda environment. First run

    source activate idtrackerai-environment

then install the library running.

    conda install -c achennu libdc1394

### Uninstall and remove software

*coming soon*

## Open idtracker.ai

If the installation succeed correctly you can test the system by launching the GUI.
Open a terminal and activate the conda environment idtrackerai-environment

    source activate idtrackerai-environment

Once the environment is activate launch the GUI

    idtrackeraiGUI

The GUI can also be launched from its main script. First go the the gui folder:

    cd idtrackerai/gui/
    
Then run the script idtrackeraiApp.py using Python.

    python idtrackeraiApp.py
    
Go to the [Quick start](http://idtracker.ai/quickstart.html) and follow the instructions
to track a simple example video.

## Documentation and examples of tracked videos

http://idtracker.ai

## Contributors
* Mattia G. Bergomi
* Francisco Romero-Ferrero
* Francisco J.H. Heras

## License
This file is part of idtracker.ai a multiple animals tracking system
described in [1].
Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
Champalimaud Foundation.

idtracker.ai is free software (both as in freedom and as in free beer):
you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.
In addition, the authors chose to distribute it free of charge by making it
publicly available (https://gitlab.com/polavieja_lab/idtrackerai.git).

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details. In addition, we require
derivatives or applications to acknowledge the authors by citing [1].

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

For more information please send an email (idtrackerai@gmail.com) or
use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.

**[1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., De Polavieja, G.G.,
(2018). idtracker.ai: Tracking all individuals in large collectives of unmarked animals (submitted).**

*F.R.-F. and M.G.B. contributed equally to this work. Correspondence should be addressed to G.G.d.P:
gonzalo.polavieja@neuro.fchampalimaud.org.*
