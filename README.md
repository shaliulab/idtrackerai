# idtracker.ai (v2.0.0-alpha)

[idtracker.ai in arXiv](https://arxiv.org/abs/1803.04351)

## NEW in v2.0.0-alpha

In this new version we made an effort to migrate the software from Python 2.7 to Python 3.6. Moreover, we also updated some libraries that are core for the system.
* Tensorflow 1.2.0 -> Tensorflow 1.9.0
* Kivy 1.9 -> Kivy 1.10
* OpenCV 2.13 -> OpenCV 3.4.2

Also, we have made the system more robust allowing to track single individuals and groups under more general conditions:

* Tracking of a single individual (skips the core of idtracker.ai)
* Tracking of individuals in groups where animals do not cross/touch/interact or do it not very frequently.
* Tracking groups without keeping the identities.
* Allow for several tracking intervals
* Create new trajectories_wo_gaps.npy file when identities are corrected in the Global Validation tab. This feature is only available when correcting the identities in the option "With animals not identified during crossings". The identity during the crossings is automatically interpolated.

## Hardware requirements

idtracker.ai has been tested in computers with the following specifications:

- Operating system: 64bit GNU/linux Mint 18.3
- CPU: Core(TM) i7-7700K CPU @4.20GHz 6 core Intel(R) or Core(TM) i7-6800K CPU @3.40GHz 4 core
- GPU: Nvidia TITAN X or GeForce GTX 1080 Ti
- RAM: 32Gb (for small groups) or 128Gb (for large groups)
- Disk: 1TB SSD

idtracker.ai is coded in Python and uses Tensorflow libraries. Due to the intense use of deep neural networks,
we recommend using a computer with a dedicated NVIDA GPU supporting compute capability 3.0 or higher.
Note that the parts of the algorithm using Tensorflow libraries will run faster with a GPU.

## Installation (v2.0.0-alpha).

The installation of idtracker.ai requires some amount of interaction with the linux
terminal. Read the following paragraph only if your are not familiar with the terminal in linux operating systems.

In Linux Mint you can open a terminal using the icon with the gray symbol ">_" on the left in the bottom bar. We provide the commands needed to install idtracker.ai from the terminal. In this documentation inputs to the terminal and outputs are shown inside of a box. You can type them directly in the command line and press ENTER to execute them.
Right-click with your mouse to copy and paste commands from the instructions to the terminal.
(NOTE: do not use the shortcut Ctrl+C and Ctrl+V as they do not work in the terminal)

The time needed to install the system varies with the output of the pre-installation checks and the download speed of the network when cloning the repository and dowloading the dependencies. In our computers and network, the total installation time is typically of <15 minutes.

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

If the command *nvidia-smi* does not show an output similar to this one, iit s possible
that the NVIDIA drivers are not installed. In Linux Mint 18.3 you can install the NVIDIA
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

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

and give it executable permissions.

    chmod u+x Miniconda3-latest-Linux-x86_64.sh

Execute the file

    ./Miniconda3-latest-Linux-x86_64.sh

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

    wget https://gitlab.com/polavieja_lab/idtrackerai/raw/2.0.0-alpha/install.sh

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

    wget https://gitlab.com/polavieja_lab/idtrackerai/raw/2.0.0-alpha/env-mint18.3-tf1.9-ocv3.4.2-kivy1.10.yml
    conda env create -f env-mint18.3-tf1.9-ocv3.4.2-kivy1.10.yml
    source activate idtrackerai-env
    git clone https://gitlab.com/polavieja_lab/idtrackerai.git
    pip install idtrackerai/.
    garden install matplotlib

### Install as a developer

If you want to develop or modify parts of the code you might want to install idtracker.ai with the following command.

    pip install -e idtrackerai/.

### Install in Windows 10

With the migration to python3 it is not possible to install idtracker.ai in Windows 10.
We remind the user that the system has not been tested in Windows 10, so different problems might arise. We appreciate users and developers to report the possible issues they might find.

1.- Clone the repository using git bash (https://gitforwindows.org/):

    git clone https://gitlab.com/polavieja_lab/idtrackerai.git

2.- Using Anaconda Prompt (https://conda.io/docs/user-guide/install/windows.html) access the *idtrackerai* folder and run the command:

    conda env create -f env-win10-tf1.9-ocv3.4.2-kivy1.10.yml

this will install idtracker.ai with GPU support. If you want to install idtracker.ai withouth GPU support (e.g. you are tracking a single animal, or you want to track groups without identification) run the command

    conda env create -f env-win10-tf1.9_nogpu-ocv3.4.2-kivy1.10.yml

3.- Using the Anaconda Prompt and from the *idtrackerai* folder run the command:

    pip install .

If you want to make modifications in the code or you don't want to reinstall idtracker.ai everytime you update the software with the *git pull* command, you can install it as a developer by doing:

    pip install -e .

4.- Install *matplotlib* for *kivy* doing:

    garden install matplotlib

## Open idtracker.ai

If the installation succeed correctly you can test the system by launching the GUI.
Open a terminal (Anaconda Prompt in Windows machines) and activate the conda environment idtrackerai-env (idtrackerai-win or idtrackerai-win-nogpu in Windows machines)

    source activate idtrackerai-env

Once the environment is activate launch the GUI

    idtrackeraiGUI

The GUI can also be launched from its main script. First go the the gui folder:

    cd idtrackerai/gui/

Then run the script idtrackeraiApp.py using Python.

    python idtrackeraiApp.py

Go to the [Quick start](http://idtracker.ai/quickstart.html) and follow the instructions
to track a simple example video.

## Monitoring idtracker.ai

As the GUI does not include many progress indicators and some processes can be computationally demanding, we recommend to monitor the flow of the system using the terminal. Also, we typically monitor the state of the CPU and RAM memory using the command

    htop

*htop* can be installed doing:

    sudo apt install htop

We monitor the performance and the state of the GPU running the command:

    watch nvidia-smi


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
