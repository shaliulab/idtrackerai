wget https://gitlab.com/polavieja_lab/idtrackerai/raw/cuda_in_conda/env-mint18.3-tf1.7-ocv2.13-kivy1.9.yml
conda env create -f env-mint18.3-tf1.7-ocv2.13-kivy1.9.yml
source activate idtrackerai-environment
git clone https://gitlab.com/polavieja_lab/idtrackerai.git
git checkout cuda_in_conda
pip install idtrackerai/.
source activate idtrackerai-environment
garden install matplotlib
