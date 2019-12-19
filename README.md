# idtracker.ai (v3)

This is the **NEW VERSION** of the tracking software [idtracker.ai](https://idtracker.ai).

idtracker.ai is a multi-animal tracking software for laboratory conditions. This work has been published in [Nature Methods](https://www.nature.com/articles/s41592-018-0295-5?WT.feed_name=subjects_software) [1] ([pdf here](https://drive.google.com/file/d/1fYBcmH6PPlwy0AQcr4D0iS2Qd-r7xU9n/view?usp=sharing))

## What is new in idtrackerai v3?

- New Graphical User Interface (GUI) based on [Pyforms](https://pyforms.readthedocs.io/en/v4/).
- Track videos from the command line with the *terminal_mode*.
- Save your preprocessing parameters for a video and load them with the *terminal_mode*. This will allow you to track batches of videos sequentially without having to interact with the GUI.
- Change advance tracking parameters using a *local_settings.py* file.
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

## Installation.

Check a more complete version of the [installation instructions in the documentation](https://idtrackerai.readthedocs.io/en/latest/how_to_install.html).

The most stable version of idtracker.ai can be installed from the PyPI using one of the following options:

1.**GUI and GPU support (I)**: This option will install idtrackerai (this repository), the [idtrackerai-app](https://gitlab.com/polavieja_lab/idtrackerai-app) and Tensorflow 1.13.1 with GPU support. Note that you will need to have the NVIDIA drivers, CUDA 10.0 and cuDNN 7.6 for it to work properly.

    pip install idtrackerai[gui,gpu]

2.**GUI and GPU support (II)**: If you do not want to install CUDA 10.0 and cuDNN 7.6 by yourself, you can install idtracker.ai in a Conda environment and then install Tensorflow 1.13 with GPU support from the Conda package manager.

    pip install idtrackerai[gui]
    conda install tensorflow-gpu=1.13

3.**No GUI and GPU support.**: Use this option if you are installing idtrackerai in a computer where you plan to run it only from the terminal (see how to do this below).

    pip install idtrackerai[gpu]

If you don't want to install CUDA 10.0 and cuDNN by yourself, install idtrackerai insider of a conda environment and then install Tensorflow 1.13 with GPU support from the Conda package manager.

    pip install idtrackerai
    conda install tensorflow-gpu=1.13

4.**GUI and no GPU support**: Use this option if you only want to use the GUI to save *.json* parameters files, or if you want to track animals using the "track without identities" feature. In this cases you don't need the GPU.

    pip install idtrackerai[gui]


5.**no GUI and no GPU support**: Use this option if you want to use idtracker.ai only to manipulate idtracker.ai objects or as an add on to another project:

    pip install idtrackerai


## Test the installation.

Once idtracker.ai is installed, you can test the installation running one of the following options.

1.**GPU support**: If you installed it using any of the GPU support options, then run:

    idtrackerai_test

2.**No GPU support**: If you installed it using the no GPU option, then run:

    idtrackerai_test --no_identities

This test will donwload a example video of around 500Mb and will execute idtracker.ai with default values for the parameters. To save the video and the results of th test in a specific folder add the following option to the command.

    idtrackerai -o path/to/folder/where/to/save/results

## Installation for developers.

1.- Clone the repository and give it a name other than *idtrackerai*. In Windows, run this step in the Git Shell.

    git clone https://gitlab.com/polavieja_lab/idtrackerai.git idtrackerai_dev

2.- Initialize all the submodules. In Windows, run this step in the Git Shell.
    
    cd idtrackerai_dev 
    git submodule update --init --recursive
    
3.- Create a new conda environment and activate it. In Windows, run the following steps in the Anaconda Prompt terminal.

    conda create -n idtrackerai_dev python=3.6
    conda activate idtrackerai_dev 
       
4.- Execute the dev_install.sh file

    sh dev_install.sh
    
5.- Install tensorflow-gpu if required 

    conda install tensorflow-gpu=1.13

## Open or run idtracker.ai

If you installed idtracker.ai with GUI support, you can run the following commands to start the GUI.

    idtrackerai

If you installed idtracker.ai without the GUI support but want to launch it from the terminal, you can run the following commands

    idtrackerai terminal_mode --load your-parameters-file.json --exec track_video

Go to the [Quick start](https://idtrackerai.readthedocs.io/en/latest/quickstart.html) and follow the instructions to track a simple example video or to learnt to save the preprocessing parameters to a *.json* file.

## Documentation and examples of tracked videos

https://idtrackerai.readthedocs.io/en/latest/index.html

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
