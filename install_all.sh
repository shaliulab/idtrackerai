#! /bin/bash

source ~/.bashrc_conda && conda activate idtrackerai4
pyinstall () {
    rm -rf build/ dist/ *egg-info && python setup.py install && rm -rf build/ dist/ *egg-info
}


pip uninstall idtrackerai
pip uninstall idtrackeri-app

cd /Users/Antonio/idtrackerai4 
pyinstall
cd /Users/Antonio/idtrackerai4/idtrackerai-app
pyinstall
cd /Users/Antonio/idtrackerai4
