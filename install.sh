wget https://gitlab.com/polavieja_lab/idtrackerai/raw/2.0.0-alpha/env-mint18.3-tf1.2-ocv2.13-kivy1.9.yml
conda env create -f env-mint18.3-tf1.9-ocv3.4.2-kivy1.10.yml
source activate idtrackerai_py3
git clone https://gitlab.com/polavieja_lab/idtrackerai.git
pip install idtrackerai/.
garden install matplotlib
