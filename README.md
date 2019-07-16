# idtracker.ai (v3.0.5-alpha)

This is the **NEW VERSION** of the tracking software idtracker.ai.

idtracker.ai is a multi-animal tracking software for laboratory conditions. This work has been published in [Nature Methods](https://www.nature.com/articles/s41592-018-0295-5?WT.feed_name=subjects_software) [1] ([pdf here](https://drive.google.com/file/d/1fYBcmH6PPlwy0AQcr4D0iS2Qd-r7xU9n/view?usp=sharing))

## What is new in version 3.0.5?

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

idtracker.ai (v3) has been tested in computers with the following specifications:

- Operating system: 64bit GNU/linux Mint 19.1 and Ubuntu 18.4
- CPU: Core(TM) i7-7700K CPU @4.20GHz 6 core Intel(R) or Core(TM) i7-6800K CPU @3.40GHz 4 core
- GPU: Nvidia TITAN X or GeForce GTX 1080 Ti
- RAM: 32Gb-128Gb (depending on the needs of the video).
- Disk: 1TB SSD

idtracker.ai is coded in python 3.6 and uses Tensorflow libraries
(version 1.13). Due to the intense use of deep neural networks, we recommend using a computer with a dedicated NVIDA GPU supporting compute capability 3.0 or higher. Note that the parts of the algorithm using Tensorflow libraries will run faster with a GPU.

## Installation (v3.0.5-alpha).

This version can be installed using the Python package manager PyPI. For an easy
installation in clusters, the application and the graphical user interface (GUI)
can be installed separately. Below we give installation instructions
for the different usage cases.


### Install NVIDIA drivers 410 (for Linux with GPU support)

This new version of idtracker.ai runs on Tensorflow 1.13.1 which requires CUDA toolkit 10 and cuDNN 7.5. In Linux, the CUDA toolkit 10.0 requires that at least the NVIDIA drivers version 410.48 are installed. You can check the CUDA compatibilities [here](https://docs.nvidia.com/deploy/cuda-compatibility/)

To check whether the NVIDIA drivers are correclty installed in your computer, open a terminal and type

    nvidia-smi

You should get an output similar to this one

    Tue Jul 16 11:45:05 2019       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 410.104      Driver Version: 410.104      CUDA Version: 10.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  GeForce GTX 1070    Off  | 00000000:01:00.0  On |                  N/A |
    | N/A   52C    P0    23W /  N/A |    373MiB /  8111MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |    0      1045      G   /usr/lib/xorg/Xorg                           214MiB |
    |    0      2499      G   ...nced GL_KHR_blend_equation_advanced_coh   106MiB |
    |    0     19703      G   ...-token=15281238DF27105D2E5E5FF8BF106888    50MiB |
    +-----------------------------------------------------------------------------+

If you fail to get this output, you can check the [installation instructions for the NVIDIA drivers 410 in this web page](https://www.mvps.net/docs/install-nvidia-drivers-ubuntu-18-04-lts-bionic-beaver-linux/)

### Installation instructions (Conda environment).

#### Preparing a Conda environment where idtracker.ai will be installed
It is good practice to install python packages in virtual environments. In particular,
we recommend using Conda virtual environments. Find here the [Conda installation
instructions for Linux, Windows and MacOS](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

When deciding whether to install Anaconda or Miniconda, you can find some information about the difference
[here](https://stackoverflow.com/questions/45421163/anaconda-vs-miniconda). For simplicity, we recommend
installing Miniconda.

To check whether the Conda package manager is installed, you can open your terminal (Anaconda prompt in Windows)
and type

    conda

if you get the following output

    conda: command not found

miniconda is not installed in your system. Follow the instructions in the link above to install it.

Create a Conda environment where idtarcker.ai will be installed.

    conda create -n idtrackerai python=3.6

You can learn more about managing Conda environments in [this link](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

Once the Conda environment has been create you should be able to activate doing

    conda activate idtrackerai

or

    source activate idtrackerai.

The following commands are to be run inside of the *idtrackerai* conda environment that you just created

#### Installation option 1 (GUI, GPU support) (NVIDIA drivers 410 already installed for Linux).

You can install idtracker.ai with GUI support with the following command (run inside the conda environment)

    pip instal idtrackerai[gui]

To get GPU support without having to manually install the CUDA 10.0 and the cuDNN 7.6, you can install Tensorflow with GPU support with the Conda package manager with the following command:

    conda install tensorflow-gpu=1.13

Conda will install the CUDA 10.0 and cuDNN 7.6 in your Conda environment for you.

#### Installation option 2 (GUI, GPU support) (NVIDIA drivers 410, CUDA 10.0 and cuDNN 7.5.0 already installed).

If you prefer to install the CUDA 10.0 and the cuDNN 7.6 in your system, you can [follow these instructions](https://medium.com/better-programming/install-tensorflow-1-13-on-ubuntu-18-04-with-gpu-support-239b36d29070) until step 6.

Then, you can install idtracker.ai with GUI an GPU support running the command:

    pip install idtrackerai[gui,gpu]

This command will install Tensorflow 1.13.1 with GPU support for you.

#### Installation option 3 (no-GUI, GPU support).

In some cases, users might want to use idtracker.ai from the command line and read the pre-processing parameters from a *.json* file (see instructions to generate a *.json* file below). This can be useful if you have a dedicated computer for tracking multiple videos in a batch and you access it with SSH, or if your are gonna install idtracker.ai in a cluster.

If the CUDA 10.0 and the cuDNN are already installed in your computer, you only need to run the following command

    pip install idtrackerai[gpu]

if you want that Conda installs the CUDA 10.0 and cuDNN 7.6 in your Conda environment, then run

    pip install idtrackerai
    conda install tensorflow-gpu=1.13

#### Installation option 4 (GUI, no-GPU support).

In some cases, the user might not need the GPU support for idtracker.ai. For example, when tracking single animals, tracking without identities, or when setting the preprocessing parameters to then track the video in a different computer or in a cluster.

In this case, you only need to install idtracker.ai with GUI support with the command

    pip install idtrackerai[gui]

### Installation instructions (Docker).

*Coming soon*

### Troubleshooting installation

*coming soon*

### Uninstall and remove software

As idtracker.ai can be now installed using a PyPI, to uninstall it you just need to execute

    pip uninstall idtrackerai

If you installed idtracker.ai inside of a Conda environment, you can also remove the environment by doing

    conda remove -n name-of-the-environment --all

## Open or run idtracker.ai

If you installed idtracker.ai with GUI support, you can run the following commands to
start the GUI.

    conda activate idtrackerai
    idtrackerai

If you installed idtracker.ai without the GUI support but want to launch it from the terminal, you can run the following commands

    conda activate idtrackerai
    idtrackerai terminal_mode --load your-parameters-file.json --exec track_video

Go to the [Quick start](http://idtracker.ai/quickstart.html) and follow the instructions to track a simple example video or to learnt to save the preprocessing parameters to a *.json* file.

## Documentation and examples of tracked videos

http://idtracker.ai

## Contributors
* Francisco Romero-Ferrero
* Mattia G. Bergomi
* Ricardo Ribeiro
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
