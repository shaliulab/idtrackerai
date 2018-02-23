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
