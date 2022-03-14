Installation and requirements
=============================

Requirements
------------
idtracker.ai v4 has been tested in computers with the following specifications:

- Operating system: 64bit GNU/linux Mint 19.1/20.2, Ubuntu 18.4 and Windows 10.
- CPU: Core(TM) i7-7700K CPU @4.20GHz 6 core Intel(R) or Core(TM) i7-6800K CPU @3.40GHz 4 core
- GPU: Nvidia TITAN X, GeForce GTX 1080 Ti, GeForce GTX 1060, 1070 and 1080.
- RAM: 16Gb-128Gb.
- Disk: 1TB SSD

idtracker.ai is coded in python 3.7 and uses PyTorch libraries and OpenCV 
(version 3).
Due to the intense use of deep neural networks, we recommend using a computer 
with a dedicated NVIDA GPU supporting compute capability 3.0 or higher.
Note that the parts of the algorithm using Tensorflow libraries will run 
faster with a GPU.


Pre-installation checks
-----------------------

**Install NVIDIA drivers +410.38 (for the installation with GPU support)**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install idtracker.ai with GPU support in your computer if you want to track 
videos keeping the identities of each animal.
Note that idtracker.ai allows users to track single animals and to track 
groups of animals without keeping the identity.
For these cases you do not need GPU support (see the Option 3 in the 
installation instructions below).

idtracker.ai v4 has been tested on PyTorch 1.10 and cudatoolkit 10.2 and 11.3.
Before installing check which NVIDIA driver you have installed and its
compatibility with the corresponding CUDA toolkit version 
(see `cuda compatiblity <https://docs.nvidia.com/deploy/cuda-compatibility/>`).

Below we give instructions to check your NVIDIA driver version and how to 
install a compatible version with CUDA 10.2 or 11.3.

**For Linux users**
*******************

To check whether the NVIDIA drivers are correctly installed in your computer, 
open a terminal and type:

.. code-block:: bash

    nvidia-smi

You should get an output similar to this one

.. code-block:: bash

    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 495.44       Driver Version: 495.44       CUDA Version: 11.5     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
    | N/A   56C    P8     5W /  N/A |    167MiB /  8111MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A      1325      G   /usr/lib/xorg/Xorg                 87MiB |
    |    0   N/A  N/A      2898      G   ...AAAAAAAAA= --shared-files       77MiB |
    +-----------------------------------------------------------------------------+


Check that in the part where it says "Driver Version" you have value higher 
than 440.33 (compatible with CUDA 10.2) or 450.80.02 (compatible with CUDA 11.3).


If you fail to get this output or your version is smaller than 440.33, 
then you will need to instal or update your nvidia drivers.

> NOTE: `this link <https://www.cyberciti.biz/faq/ubuntu-linux-install-nvidia-driver-latest-proprietary-driver/>`
> has nice instructions to get the latest NVIDIA drivers either using your Update Manager or the terminal.

1. Clean the system of other Nvidia drivers

.. code-block:: bash

    sudo apt-get purge nvidia*

2. Check which is the latest driver version system in `this link <https://www.nvidia.com/object/unix.html>`_.

3. Update and upgrade your system:

.. code-block:: bash

    sudo apt update
    sudo apt upgrade

1. Check which is the latest available version of the NVIDIA drivers for your system:

.. code-block:: bash

    apt search nvidia-driver

5. Install the NVIDIA GPU driver. In the following command, substitute the XXX by the number of the driver you want to install (e.g. `nvidia-driver-495`).

.. code-block:: bash

    sudo apt-get install nvidia-driver-XXX

6. Reboot the system.

.. code-block:: bash

    sudo reboot

7. Check the installation.

.. code-block:: bash

    nvidia-smi

**For Windows users**
*********************

To check which NVIDIA drivers you have installed in your computer following these steps
(adapted from `this page <https://www.drivereasy.com/knowledge/how-to-check-nvidia-driver-version-easily/>`_):

1. Right click any empty area on your desktop screen, and select NVIDIA Control Panel.

2. Click System Information (on the bottom left corner) to open the driver information.

3. Check the Driver version in the Details section.

You can download the latest driver available for your GPU from `the NVIDIA webpage <https://www.nvidia.com/Download/index.aspx>`_.

After downloading the *.exe* file, execute it and follow the instructions.
After the installation you will be asked to reboot the computer, please do so for the installation to be complete.

> NOTE: For Windows you will need an NVIDIA driver >=441.22 for CUDA 10.2 and >=456.38 for CUDA 11.3.

**Preparing a Conda environment (for Linux and Windows)**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is good practice to install python packages in virtual environments. In particular,
we recommend using Conda virtual environments. Find here the `Conda installation
instructions for Linux and Windows <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_.

When deciding whether to install Anaconda or Miniconda, you can find some information about the differences
`here <https://stackoverflow.com/questions/45421163/anaconda-vs-miniconda>`_. For simplicity, we recommend
installing Miniconda.

From now on, every time we refer to the *terminal*, Linux users are meant to use the command line and Windows user
are meant to use the Anaconda Powershell Prompt that it is installed when installing Miniconda or Anaconda.

To check whether the Conda package manager is installed, you can open a terminal and type

.. code-block:: bash

    conda

if you get the following output

.. code-block:: bash

    conda: command not found

Miniconda is not installed in your system. Follow the instructions in the link above to install it.

Create a Conda environment where idtarcker.ai will be installed.

.. code-block:: bash

    conda create -n idtrackerai python=3.7

You can learn more about managing Conda environments in
`this link <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

Once the Conda environment has been create you should be able to activate it doing

.. code-block:: bash

    conda activate idtrackerai

or

.. code-block:: bash

    source activate idtrackerai.


**Installation**
~~~~~~~~~~~~~~~~

Assuming that you have the latest version of the NVIDIA drivers installed, and 
Anaconda or Miniconda installed, the recomended way to install 
idtracker.ai v4 is using the following commands (to be run in a linux 
terminal or in the Anaconda Powershell Prompt in Windows):

.. code-block:: bash

    conda create -n idtrackerai python=3.7
    conda activate idtrackerai
    pip install idtrackerai[gui]
    conda install pytorch torchvision -c pytorch

Below we give more detailed installation instructions for the different usage 
scenarios.

**Option 1 (GUI, GPU support) (NVIDIA drivers already installed)**
********************************************************************************

Once you have created and activated the conda environment, 
you can install idtracker.ai with GUI support with the following command

.. code-block:: bash

    pip install idtrackerai[gui]

To get GPU support without having to manually install the CUDA 10.2 or 11.3,
you can install PyTorch with GPU support from the Conda package manager with the following command:

.. code-block:: bash

    conda install pytorch torchvision -c pytorch 

This will install the latest version of cudatoolkit. To specify a lower version 
use the command:

.. code-block:: bash

    conda install pytorch torchvision cudatoolkit=10.2 -c pytorch 


**Option 2 (GUI, GPU support) (NVIDIA drivers and CUDA already installed)**
*************************************************************************************************

If you have already installed CUDA system-wide, then you can install 
idtracker.ai with GUI an GPU support running the command:

.. code-block:: bash

    pip install idtrackerai[gui,gpu]

This will install the latest version of `pytorch` and `torchvision` using PyPI 
instead of conda.

**Option 3 (GUI, no-GPU support)**
**********************************

In some cases, you might not need the GPU support for idtracker.ai.
For example, when tracking single animals, tracking animals without keeping the 
identities along the video, or when setting the preprocessing parameters to 
then track the video in a different computer or in a cluster.

In this case, you only need to install idtracker.ai with GUI support with the 
command:

.. code-block:: bash

    pip install idtrackerai[gui]

**Option 4 (no-GUI, GPU support)**
**********************************

You might want to use idtracker.ai from the command line and read the pre-processing
parameters from a *.json* file (see instructions to generate a *.json* file in
the :doc:`tracking_from_terminal` section). This can be useful if you have a
dedicated computer for tracking multiple videos in batches and you access it with SSH,
or if your are going to install idtracker.ai in a cluster.

If CUDA is are already installed in your computer system-wide,
you only need to run the following command:

.. code-block:: bash

    pip install idtrackerai[cli, gpu]

If you want Conda to install the CUDA in your Conda environment, then run

.. code-block:: bash

    pip install idtrackerai[cli]
    conda install pytorch torchvision -c pytorch 

This will install the latest version of cudatoolkit. To specify a lower version 
use the command:

.. code-block:: bash

    conda install pytorch torchvision cudatoolkit=10.2 -c pytorch 


**Option 4 (no-GUI, no-GPU support)**
*************************************

Some times you might want to install idtrackerai in an environment so that you
can manipulate and open idtracker.ai files. For that you just need to run 
the command:

.. code-block:: bash

    pip install idtrackerai

Note that with this installation mode, you won't have any CLI or GUI to track 
videos.


**Uninstall and remove the software**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As idtracker.ai can be now installed using a PyPI, to uninstall it you just 
need to execute

.. code-block:: bash

    pip uninstall idtrackerai

If you installed idtracker.ai inside of a Conda environment, you can 
also remove the environment by doing

.. code-block:: bash

    conda remove -n name-of-the-environment --all
