wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
chmod +x Miniconda2-latest-Linux-x86_64.sh
./Miniconda2-latest-Linux-x86_64.sh

source ~/.bashrc

wget https://gitlab.com/polavieja_lab/idtrackerai/raw/write_setup/environment-mint18.1.yml
conda env create -f environment-mint18.1.yml
source activate idtrackerai-environment
git clone https://gitlab.com/polavieja_lab/idtrackerai.git
pip install idtrackerai/.
source activate idtrackerai-environment
garden install matplotlib

