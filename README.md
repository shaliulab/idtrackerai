# idtracker.ai (v3.0.0-alpha)

This is the **NEW VERSION** of the tracking software idtracker.ai.

idtracker.ai is a multi-animal tracking software for laboratory conditions. This
work has been published in [Nature Methods](https://www.nature.com/articles/s41592-018-0295-5?WT.feed_name=subjects_software) [1] ([pdf here](https://drive.google.com/file/d/1fYBcmH6PPlwy0AQcr4D0iS2Qd-r7xU9n/view?usp=sharing))

## What is new in version 3.0.0?

- New Graphical User Interface (GUI) based on [Pyforms](https://pyforms.readthedocs.io/en/v4/).
- Track videos from the command line with the *terminal_mode*.
- Save your preprocessing parameters for a video and load them with the *terminal_mode*. This will allow you to track batches of videos sequentially without having to interact with the GUI.
- Change advance tracking parameters using a *local_settins.py* file.
- Improved memory management during tracking. Identification images and sets of pixels can be
now saved in RAM or DISK. Set these parameters using the *local_settings.py* file. Saving images and pixels in the DISK will make the tracking slower, but will allow you to track longer videos with less RAM memory.
- Improved data storage management. Use the parameter *DATA_POLICY* in the *local_settings.py* file to decide which files to save after the tracking. For example, this will prevent you from storing heavy unnecessary files if what you only need are the trajectories.
- Improved validation and correction of trajectories with a new GUI based on [Python Video Annotator](https://pythonvideoannotator.readthedocs.io/en/master/).
- Overall improvements in the internal structure of the code.
- Multiple bugs fixed.

## Hardware requirements

idtracker.ai has been tested in computers with the following specifications:

- Operating system: 64bit GNU/linux Mint 18.3
- CPU: Core(TM) i7-7700K CPU @4.20GHz 6 core Intel(R) or Core(TM) i7-6800K CPU @3.40GHz 4 core
- GPU: Nvidia TITAN X or GeForce GTX 1080 Ti
- RAM: 32Gb-128Gb (depending on the needs of the video).
- Disk: 1TB SSD

idtracker.ai is coded in python 3.6 and uses Tensorflow libraries
(version 1.13). Due to the intense use of deep neural networks, we recommend using a computer with a dedicated NVIDA GPU supporting compute capability 3.0 or higher. Note that the parts of the algorithm using Tensorflow libraries will run faster with a GPU.

## Installation (v3.0.0-alpha).

The installation of idtracker.ai requires some amount of interaction with the linux
terminal. Read the following paragraph only if your are not familiar with the terminal in linux operating systems.

In Linux Mint you can open a terminal using the icon with the gray symbol ">_" on the left in the bottom bar. We provide the commands needed to install idtracker.ai from the terminal. In this documentation inputs to the terminal and outputs are shown inside of a box. You can type them directly in the command line and press ENTER to execute them.
Right-click with your mouse to copy and paste commands from the instructions to the terminal.
(NOTE: do not use the shortcut Ctrl+C and Ctrl+V as they do not work in the terminal)

The time needed to install the system varies with the output of the pre-installation checks and the download speed of the network when cloning the repository and dowloading the dependencies. In our computers and network, the total installation time is typicall of 15 minutes.

### Pre-installation checks

###### Make sure that your system is updated

Check if your system is up to date running the following command in the terminal:

    sudo apt update

Upgrade your system running:

    sudo apt upgrade

###### GPU Drivers.

Make sure that your GPU drivers are installed.

If you are using an NVIDIA GPU you can check that the drivers are properly
installed typing.

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

If the command *nvidia-smi* does not show an output similar to this one, is possible
thay the NVIDIA drivers are not installed. In Linux Mint 18.3 you can install the NVIDIA
drivers using the Driver Manager that can be found in the Menu button in the bottom left of the screen.

**NOTE** Do not install the *intel-microcode* driver
as it might enter in conflict with the parts of idtracker.ai that are parallelized. When installing this driver we have experienced hangs in the background subtraction and in the segmentation parts of the algorithm.

###### Miniconda package manager

The installation process requires [miniconda](https://conda.io/miniconda.html) to be installed in your computer. Skip the next paragraphs if Miniconda2 or Miniconda3 are already installed.

To check whether miniconda is installed in your computer type

    conda

if you get the following output

    conda: command not found

miniconda is not installed in your system. Follow the next instructions to install it.

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

**IMPORTANT** At the end of the installation close the terminal and open a new one.

### Installation

Using the terminal, download the file [install.sh](https://gitlab.com/polavieja_lab/idtrackerai/raw/cuda_in_conda/install.sh)
using the following command.

    wget https://gitlab.com/polavieja_lab/idtrackerai/raw/1.0.2-alpha/install.sh

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

    wget https://gitlab.com/polavieja_lab/idtrackerai/raw/1.0.3-alpha/env-mint18.3-tf1.2-ocv2.13-kivy1.9.yml
    conda env create -f env-mint18.3-tf1.2-ocv2.13-kivy1.9.yml
    source activate idtrackerai-environment
    git clone https://gitlab.com/polavieja_lab/idtrackerai.git
    pip install idtrackerai/.
    garden install matplotlib

### Troubleshooting installation

*coming soon*

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

**[1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., de Polavieja, G.G., Nature Methods, 2019.
idtracker.ai: tracking all individuals in small or large collectives of unmarked animals.
(F.R.-F. and M.G.B. contributed equally to this work.
Correspondence should be addressed to G.G.d.P: gonzalo.polavieja@neuro.fchampalimaud.org)**

*F.R.-F. and M.G.B. contributed equally to this work. Correspondence should be addressed to G.G.d.P:
gonzalo.polavieja@neuro.fchampalimaud.org.*
