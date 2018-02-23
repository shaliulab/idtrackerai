mkdir tmp
cd tmp
wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
chmod +x Miniconda2-latest-Linux-x86_64.sh
./Miniconda2-latest-Linux-x86_64.sh
source ~/.bashrc
# wget environment-mint18.1.yml
conda env create -f environment-mint18.1.yml
source activate idtrackerai-environment
# git clone
# pip install idtrackerai
#chmod +x idtrackerai/run.sh
