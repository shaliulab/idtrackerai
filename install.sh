wget https://gitlab.com/polavieja_lab/idtrackerai/raw/1.0.2-alpha/env-mint18.3-tf1.2-ocv2.13-kivy1.9.yml
conda env create -f env-mint18.3-tf1.2-ocv2.13-kivy1.9.yml
source activate idtrackerai-environment
git clone https://gitlab.com/polavieja_lab/idtrackerai.git
pip install idtrackerai/.
garden install matplotlib
