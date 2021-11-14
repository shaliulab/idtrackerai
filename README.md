# idtracker.ai (v4)

[idtracker.ai](https://idtracker.ai) is a multi-animal tracking software for 
laboratory conditions. 
This work has been published in 
[Nature Methods](https://www.nature.com/articles/s41592-018-0295-5?WT.feed_name=subjects_software) 
[1] ([pdf here](https://drive.google.com/file/d/1fYBcmH6PPlwy0AQcr4D0iS2Qd-r7xU9n/view?usp=sharing))

## What is new in idtrackerai v4?

- Works with Python 3.7.
- Remove Kivy submodules and stop support for old Kivy GUI.
- Neural network training is done with Pytorch 1.10.0.
- Identification images are saved as uint 8.
- Crossing detector images are the same as the identification images. This
 saves computing time and makes the process of generating the images faster.
- Improve data pipeline for the crossing detector.
- Parallel saving and loading of identification images (only for Linux)
- Simplify code for connecting blobs from frame to frame.
- Remove unnecessary execution of the blobs connection algorithm.
- Background subtraction considers the ROI
- Allows to save trajectories as csv with the advanced parameter 
`CONVERT_TRAJECTORIES_DICT_TO_CSV_AND_JSON` (using the 
`local_settings.py` file).
- Allows to change the output width (and height) of the individual-centered 
videos with the advanced parameter `INDIVIDUAL_VIDEO_WIDTH_HEIGHT` 
(using the `local_settings.py` file).
- Horizontal layout for graphical user interface (GUI). This layout can be
deactivated using the `local_settings.py` setting  `NEW_GUI_LAYOUT=False`.
- Width and height of GUI can be changed using the `local_settings.py` using 
the `GUI_MINIMUM_HEIGHT` and `GUI_MINIMUM_WIDTH` variables.
- Add ground truth button to validation GUI.
- Added "Add setup points" featrue to store landmark points in the video frame 
that will be stored in the `trajectories.npy` and `trajectories_wo_gaps.npy` 
in the key `setup_poitns`. Users can use this points to perform behavioural
analysis that requires landmarks of the experimental setup.
- Improved code formatting using the black formatter.
- Better factorization of the TrackerApi.
- Some bugs fixed.
- Better documentation of main idtracker.ai objects (`video`, `blob`, 
`list_of_blobs`, `fragment`, `list_of_fragments`, 
`global_fragment` and `list_of_global_fragments`)
- Dropped support for MacOS

## Hardware requirements

idtracker.ai (v4) has been tested in computers with the following
 specifications:

- Operating system: 64bit GNU/linux Mint 19.1 and Ubuntu 18.4
- CPU: Core(TM) i7-7700K CPU @4.20GHz 6 core Intel(R) or Core(TM) i7-6800K 
CPU @3.40GHz 4 core
- GPU: Nvidia TITAN X or GeForce GTX 1080 Ti
- RAM: 32Gb-128Gb (depending on the needs of the video).
- Disk: 1TB SSD

idtracker.ai is coded in python 3.7 and uses Pytorch libraries
(version 1.10.0). 
Due to the intense use of deep neural networks, we recommend using a
 computer with a dedicated NVIDA GPU supporting compute capability 3.0 
 or higher. 
Note that the parts of the algorithm using Tensorflow libraries will run
 faster with a GPU.

## Installation.

Frist of all, make sure that you have the latest version of the CUDA driver 
installed (currenly tested with 495.44)

The recomended way to install idtracker.ai v4 is using the following commands:

    conda create -n idtrackerai python=3.7
    pip install idtrackerai[gui] --pre
    conda install pytorch torchvision -c pytorch

This will install the latest version of pytorch (1.10.0) and torchvision (0.11.1)
and the cudatoolkit (version 11.3.1).

> NOTE: You can install a lower version of the cudatoolkit using the command
> `conda isntall pytorch torchvision cudatoolkit=10.2 -c pytorch`

> NOTE: If your computer does not have support for GPU computing, then install 
pytorch with the `cpuonly` mode activated. So, you just need to change the
last line by: `conda install pytorch torchvision cpuonly -c pytorch`

> NOTE: Check a more complete version of the installation instructions 
> in the [documentation](https://idtrackerai.readthedocs.io/en/latest/how_to_install.html).

## Test the installation.

Once idtracker.ai is installed, you can test the installation running one of
 the following options.

1.**GPU support**: If you installed it using any of the GPU support options
, then run:

    idtrackerai_test

2.**No GPU support**: If you installed it using the no GPU option, then run:

    idtrackerai_test --no_identities

## Installation for developers.

1.- Clone the repository. In Windows, run this step in the Git Shell:

    git clone https://gitlab.com/polavieja_lab/idtrackerai.git idtrackerai_dev

2.- Initialize all the submodules. In Windows, run this step in the Git Shell:
    
    cd idtrackerai_dev 
    git checkout v4-dev
    git submodule update --init --recursive
    
3.- Create a conda environment using the `dev-environment.yml` file
 and activate it. In Windows, run the following steps in the Anaconda Prompt 
 terminal:

    conda env create -f dev-environment.yml python=3.7
    conda activate idtrackerai_dev 
       
4.- Execute the dev_install.sh file:

    sh dev_install.sh

## Open or run idtracker.ai

To run idtracker.ai just execute the following command inside of the 
corresponding conda environment:

    idtrackerai

If you want to execute idtracker.ai using the `terminal_mode` and loading a 
`.json` file where the parameters are stored using the following command:

    idtrackerai terminal_mode --load your-parameters-file.json --exec track_video

Go to the 
[Quick start](https://idtrackerai.readthedocs.io/en/latest/quickstart.html)  
 and follow the instructions to track a simple example video and learn to
  save the preprocessing parameters to a `.json` file.

  
## Notes for delevopers

This repository contains idtracker.ai's algorithm, the repository (submodule)
[idtrackerai-app](https://gitlab.com/polavieja_lab/idtrackerai-app) contains 
the CLI and GUI to track videos using the idtracker.ai's algorithm.

The validation GUI used to check the results of the tracking is integrated 
inside of a bigger project called 
[Python-Video-Annotator](https://pythonvideoannotator.readthedocs.io/en/master/).
The idtracker.ai's validation GUI is a plugin inside of this bigger project,
but it has its own repository, the 
[pythonvideoannotator-module-idtrackerai](https://github.com/video-annotator/pythonvideoannotator-module-idtrackerai)].

We coded idtracker.ai's GUI in this way so that in the future other CLI or GUI
can be coded without affecting the idtracker.ai algorithm, or the algorithm
can be modified without affecting the current GUI and CLI.
## Documentation and examples of tracked videos

Check more information in the [idtracker.ai webpage](https://idtrackerai.readthedocs.io/en/latest/index.html)

## Contributors
* Francisco Romero-Ferrero (2015-)
* Mattia G. Bergomi (2015-2018)
* Ricardo Ribeiro (2018-2020)
* Francisco J.H. Heras (2015-)

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

**[1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., 
de Polavieja, G.G., Nature Methods, 2019.
idtracker.ai: tracking all individuals in small or large collectives 
of unmarked animals.
(F.R.-F. and M.G.B. contributed equally to this work.
Correspondence should be addressed to G.G.d.P: 
gonzalo.polavieja@neuro.fchampalimaud.org)**

*F.R.-F. and M.G.B. contributed equally to this work. 
Correspondence should be addressed to G.G.d.P:
gonzalo.polavieja@neuro.fchampalimaud.org.*
