wget https://gitlab.com/polavieja_lab/idtrackerai/raw/write_setup/environment-mint18.1.yml
conda env create -f environment-mint18.1.yml
source activate idtrackerai-environment
git clone https://gitlab.com/polavieja_lab/idtrackerai.git
pip install idtrackerai/.
source activate idtrackerai-environment
garden install matplotlib
chmod u+x idtrackerai/run.sh
