# idtrackerai
## Tracking all individuals with correct identities in large animal collectives

idtrackerai allows to track animals in small and large collectives using convolutional neural networks

### Installation in linux Mint 18.1

Download install.sh and create a folder idtrackerai.

    cd idtrackerai
    ./install.sh
    
If install.sh does not have executable permission type

    chmod u+x install.sh
    ./install.sh

install.sh will first install miniconda, you will be asked to accept the terms 
and conditions, then will download this repository and install the package.

At the end of the installation activate the environment and launch the GUI:

    conda activate idtrackera-environment
    idtrackeraiGUI

If the installation did not succeed try proceeding step by step, by running 
the following commands in your terminal:

    wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
    chmod +x Miniconda2-latest-Linux-x86_64.sh
    ./Miniconda2-latest-Linux-x86_64.sh
    source ~/.bashrc
    wget http://git.yalp.io/mattia/IdTrackerDeep/raw/write_setup/environment-mint18.1.yml
    conda env create -f environment-mint18.1.yml
    source activate idtrackerai-environment
    git clone http://git.yalp.io/mattia/IdTrackerDeep.git
    pip install IdTrackerDeep/.
    chmod +x IdTrackerDeep/run.sh
    

## Documentation and examples of tracked videos
http://idtracker.ai

## Contributors

