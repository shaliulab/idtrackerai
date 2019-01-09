# idtracker.ai (v2.0.0-alpha)

[idtracker.ai](http://idtracker.ai/) is a software that tracks animals in groups keeping the identity of every individual after they touch or cross.
[idtracker.ai in Nature Methods](http://dx.doi.org/10.1038/s41592-018-0295-5)
[idtracker.ai in arXiv](https://arxiv.org/abs/1803.04351)

## Requirements

[idtracker.ai](http://idtracker.ai/) has been developed and tested in computers with the following specifications:

- Operating systems: 64bit GNU/linux Mint 18.3
- CPU: Core(TM) i7-7700K CPU @4.20GHz 6 core Intel(R) or Core(TM) i7-6800K CPU @3.40GHz 4 core
- GPU: Nvidia TITAN X or GeForce GTX 1080 Ti
- RAM: 32Gb (for small groups) or 128Gb (for large groups)
- Disk: 1TB SSD

However, the software also works in Windows 10, in Ubuntu 18.04 and Linux Mint 19.1.

[idtracker.ai](http://idtracker.ai/) is coded in Python and uses Tensorflow. Due to the intense use of deep neural networks,
we recommend using a computer with a dedicated NVIDA GPU supporting compute capability 3.0 or higher.
Note that the parts of the algorithm using Tensorflow will run faster with a GPU.

[idtracker.ai](http://idtracker.ai/) can also be used to track single individuals or groups of animals without keeping the identities. In these cases, the system does not require the intensive use of a GPU. We provide *Conda* environments to work without a GPU.

## Installation.

The installation of idtracker.ai requires some amount of interaction with the
terminal (command line). Read the following paragraph only if your are not familiar with the terminal.

**In Linux Mint** you can open a terminal using the icon with the gray symbol **>_** on the left in the bottom bar. We provide the commands needed to install idtracker.ai from the terminal. In this documentation inputs to the terminal and outputs are shown inside of a box. You can type them directly in the command line and press ENTER to execute them.
Right-click with your mouse to copy and paste commands from the instructions to the terminal.
(NOTE: do not use the shortcut Ctrl+C and Ctrl+V as they do not work in the terminal)

**In Windows** we recommend using the *git shell* terminal to interact with the git repositories and the *Anaconda Prompt* to install and run [idtracker.ai](http://idtracker.ai/).

The time needed to install the system varies with the output of the pre-installation checks and the download speed of the network when cloning the repository and dowloading the dependencies. In our computers and network, the total installation time is typically of <15 minutes.

### Pre-installation checks

###### Make sure that your system is updated

Check if your system is up to date running the following command in the terminal:

    sudo apt update

Upgrade your system running:

    sudo apt upgrade

###### GPU Drivers.

Make sure that your GPU drivers are installed.

**In linux**, if you are using an NVIDIA GPU you can check that the drivers are properly
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

**In linux**, to check whether *miniconda* is installed in your computer type

    conda

if you get the following output

    conda: command not found

*miniconda* is not installed in your system. Follow the next instructions to install it.

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

At the end of the installation close the terminal and open a new one.

**In Windows** you will need to install the [Anaconda distribution](https://https://conda.io/docs/user-guide/install/windows.html) in order to get the Anaconda Prompt which you will use to install and run [idtracker.ai](http://idtracker.ai/).

###### Git

To install [idtracker.ai](http://idtracker.ai/) you will need to clone this repository using the *git* version control system.

**In Linux**, you can install *Git* with the following command in the terminal

    sudo apt-get install git

**In Windows**, you can get the *Git BASH* installing [git for Windows](https://gitforwindows.org/).

### Installation

##### 1. Clone this repository

**In Linux**, in the terminal (**in Windows**, in the Git BASH) run the command:

    git clone https://gitlab.com/polavieja_lab/idtrackerai.git



##### 2. Access the *idtrackerai* folder created in the previous step

**In Linux**, in the terminal (**in Windows**, in the Anaconda Prompt), run the command:

    cd idtrackerai

**In Windows**, this step might change if the default folder for the Anaconda Prompt and the Git BASH is different. Just use the command ``cd directory`` and ``cd ..`` to move around the different folders until you get to the folder *idtrackerai* created in the previous step.  

##### 3. Install the corresponding conda environment

**In Linux**, in the terminal, run the command:

    conda env create -f idtrackerai-env_linux.yml


if you want to install the version without GPU support substitute *idtrackerai-env_linux.yml* by *idtrackerai-env_linux_nogpu.yml*

**In Windows**, in the Anaconda Prompt, run the command:

    conda env create -f idtrackerai-env_win.yml

if you want to install the version without GPU support substitute *idtrackerai-env_win.yml* by *idtrackerai-env_win_nogpu.yml*

##### 4. Activate the conda environment

**In Linux**, in the terminal (**in Windows**, in the Anaconda Prompt), run the command:

    conda activate idtrackerai-env

In some installations of *miniconda* you will need to use the command

    source activate idtrackerai-env

In the Anaconda Prompt **in Windows** you can also use simply

    activate idtrackerai-env

If you have installed the version without GPU support substitute *idtrackerai-env* by *idtrackerai-nogpu*.

##### 5. Install idtracker.ai

**In Linux**, in the terminal (**in Windows**, in the Anaconda Prompt) run the command:

    pip install .

if you want to install the software with developer options, use the command:

    pip install -e .

##### 6. Install matplotlib for Kivy

**In Linux**, in the terminal (**in Windows**, in the Anaconda Prompt) run the command:

    garden install matplotlib

## Open idtracker.ai

If the installation succeed correctly you can test the system by launching the GUI.
Open a terminal in **Linux** or an Anaconda Prompt in **Windows** and activate the conda environment idtrackerai-env (idtrackerai-win or idtrackerai-win-nogpu in Windows machines).

    source activate idtrackerai-env

Depending on the configuration of your *conda*, in linux systems you can also activate the environment running

    conda activate idtrackerai-env

In the Anaconda Prompt you can activate the environment doing

    conda activate idtrackerai-env

or simply

    activate idtrackerai-env

Once the environment is activate launch the GUI

    idtrackeraiGUI

The GUI can also be launched from its main script. First go the the gui folder:

    cd idtrackerai/gui/

Then run the script idtrackeraiApp.py using Python.

    python idtrackeraiApp.py

Go to the [Quick start](http://idtracker.ai/quickstart.html) and follow the instructions
to track a simple example video.

## Monitoring idtracker.ai

As the GUI does not include many progress indicators and some processes can be computationally demanding, we recommend to monitor the flow of the system using the terminal **in Linux**. Also, we typically monitor the state of the CPU and RAM memory using the command

    htop

*htop* can be installed doing:

    sudo apt install htop

We monitor the performance and the state of the GPU running the command:

    watch nvidia-smi


## Documentation and examples of tracked videos

http://idtracker.ai

## Contributors
* Francisco Romero-Ferrero
* Mattia G. Bergomi
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
