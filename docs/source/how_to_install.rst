Installation and requirements
=============================

^^^^^^^^^^^^
Requirements
^^^^^^^^^^^^
idtracker.ai v3 has been tested in computers with the following specifications:

- Operating system: 64bit GNU/linux Mint 19.1 and Ubuntu 18.4
- CPU: Core(TM) i7-7700K CPU @4.20GHz 6 core Intel(R) or Core(TM) i7-6800K CPU @3.40GHz 4 core
- GPU: Nvidia TITAN X or GeForce GTX 1080 Ti
- RAM: 32Gb-128Gb (depending on the needs of the video).
- Disk: 1TB SSD

idtracker.ai is coded in python 3.6 and uses Tensorflow libraries
(version 1.13). Due to the intense use of deep neural networks, we recommend using a computer with a dedicated NVIDA GPU supporting compute capability 3.0 or higher. Note that the parts of the algorithm using Tensorflow libraries will run faster with a GPU.

^^^^^^^^^^^^^^^^^^^^^^^
Pre-installation checks
^^^^^^^^^^^^^^^^^^^^^^^

This version can be installed using the Python package manager PyPI. For an easy
installation in clusters, the application and the graphical user interface (GUI)
can be installed separately. Below we give installation instructions
for the different usage cases.

**Install NVIDIA drivers 410 (for Linux with GPU support)**
***********************************************************

This new version of idtracker.ai runs on Tensorflow 1.13.1 which requires CUDA toolkit 10 and cuDNN 7.5. In Linux, the CUDA toolkit 10.0 requires that at least the NVIDIA drivers version 410.48 are installed. You can check the CUDA compatibilities `here <https://docs.nvidia.com/deploy/cuda-compatibility/>`_

To check whether the NVIDIA drivers are correclty installed in your computer, open a terminal and type

.. code-block::

    nvidia-smi

You should get an output similar to this one

.. code-block::

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

If you fail to get this output, you can check the `installation instructions for the NVIDIA drivers 410 in this web page <https://www.mvps.net/docs/install-nvidia-drivers-ubuntu-18-04-lts-bionic-beaver-linux/>`_

**Preparing a Conda environment where idtracker.ai will be installed**
***********************************************************************

It is good practice to install python packages in virtual environments. In particular,
we recommend using Conda virtual environments. Find here the `Conda installation
instructions for Linux, Windows and MacOS <https://docs.conda.io/projects/conda/en/latest/user-guide/install/)>`_/

When deciding whether to install Anaconda or Miniconda, you can find some information about the difference
`here <https://stackoverflow.com/questions/45421163/anaconda-vs-miniconda>`_. For simplicity, we recommend
installing Miniconda.

To check whether the Conda package manager is installed, you can open your terminal (Anaconda prompt in Windows)
and type

.. code-block::

    conda

if you get the following output

.. code-block::

    conda: command not found

miniconda is not installed in your system. Follow the instructions in the link above to install it.

Create a Conda environment where idtarcker.ai will be installed.

.. code-block::

    conda create -n idtrackerai python=3.6

You can learn more about managing Conda environments in `this link <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

Once the Conda environment has been create you should be able to activate doing

.. code-block::

    conda activate idtrackerai

or

.. code-block::

    source activate idtrackerai.

The following commands are to be run inside of the *idtrackerai* conda environment that you just created

^^^^^^^^^^^^
Installation
^^^^^^^^^^^^

**Option 1 (GUI, GPU support) (NVIDIA drivers 410 already installed for Linux)**
********************************************************************************

You can install idtracker.ai with GUI support with the following command (run inside the conda environment)

.. code-block::

    pip instal idtrackerai[gui]

To get GPU support without having to manually install the CUDA 10.0 and the cuDNN 7.6, you can install Tensorflow with GPU support with the Conda package manager with the following command:

.. code-block::

    conda install tensorflow-gpu=1.13

Conda will install the CUDA 10.0 and cuDNN 7.6 in your Conda environment for you.

**Option 2 (GUI, GPU support) (NVIDIA drivers 410, CUDA 10.0 and cuDNN 7.5.0 already installed)**
*************************************************************************************************

If you prefer to install the CUDA 10.0 and the cuDNN 7.6 in your system, you can [follow these instructions](https://medium.com/better-programming/install-tensorflow-1-13-on-ubuntu-18-04-with-gpu-support-239b36d29070) until step 6.

Then, you can install idtracker.ai with GUI an GPU support running the command:

.. code-block::

    pip install idtrackerai[gui,gpu]

This command will install Tensorflow 1.13.1 with GPU support for you.

**Option 3 (no-GUI, GPU support)**
**********************************

In some cases, users might want to use idtracker.ai from the command line and read the pre-processing parameters from a *.json* file (see instructions to generate a *.json* file below). This can be useful if you have a dedicated computer for tracking multiple videos in a batch and you access it with SSH, or if your are gonna install idtracker.ai in a cluster.

If the CUDA 10.0 and the cuDNN are already installed in your computer, you only need to run the following command

.. code-block::

    pip install idtrackerai[gpu]

if you want that Conda installs the CUDA 10.0 and cuDNN 7.6 in your Conda environment, then run

.. code-block::

    pip install idtrackerai
    conda install tensorflow-gpu=1.13

**Option 4 (GUI, no-GPU support)**
**********************************

In some cases, the user might not need the GPU support for idtracker.ai. For example, when tracking single animals, tracking without identities, or when setting the preprocessing parameters to then track the video in a different computer or in a cluster.

In this case, you only need to install idtracker.ai with GUI support with the command

.. code-block::

    pip install idtrackerai[gui]

^^^^^^^^^^^^^^^^^^^^^^^^^^
Installation with (Docker)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*Coming soon*

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Troubleshooting installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*coming soon*

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Uninstall and remove software
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As idtracker.ai can be now installed using a PyPI, to uninstall it you just need to execute

.. code-block::

    pip uninstall idtrackerai

If you installed idtracker.ai inside of a Conda environment, you can also remove the environment by doing

.. code-block::

    conda remove -n name-of-the-environment --all
