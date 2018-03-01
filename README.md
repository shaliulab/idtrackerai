# idtrackerai
## Tracking all individuals with correct identities in large animal collectives

idtrackerai allows to track animals in small and large collectives using convolutional neural networks.

### Installation in linux Mint 18.1

Download [install.sh](install.sh) and create a folder idtrackerai.

    cd idtrackerai
    ./install.sh

If install.sh does not have executable permission type

    chmod u+x install.sh
    ./install.sh

install.sh will first install miniconda, you will be asked to accept the terms
and conditions, then will download this repository and install the package.

At the end of the installation activate the environment and launch the GUI:

    conda activate idtrackerai-environment
    idtrackeraiGUI

If the installation did not succeed try proceeding step by step, by running
the following commands in your terminal:

    wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
    chmod +x Miniconda2-latest-Linux-x86_64.sh
    ./Miniconda2-latest-Linux-x86_64.sh
    source ~/.bashrc
    wget https://gitlab.com/polavieja_lab/idtrackerai/raw/write_setup/environment-mint18.1.yml
    conda env create -f environment-mint18.1.yml
    source activate idtrackerai-environment
    git clone https://gitlab.com/polavieja_lab/idtrackerai.git
    pip install idtrackerai/.
    
    garden install matoplotlib
    chmod +x idtrackerai/run.sh



## Documentation and examples of tracked videos
http://idtracker.ai

## Contributors
* Mattia G. Bergomi
* Francisco Romero-Ferrero
* Francisco J.H. Heras

## License
This is idtracker.ai a multiple animals tracking system
described in [1].
Copyright (C) 2017- Bergomi, M.G., Romero-Ferrero, F., Heras, F.J.H.

idtracker.ai is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details. In addition, we require
derivatives or applications to acknowledge the authors by citing [1].

A copy of the GNU General Public License is available [here](LICENSE).

For more information please send an email (idtrackerai@gmail.com) or
use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.

**[1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., De Polavieja, G.G.,
(2018). idtracker.ai: Tracking all individuals with correct identities in large
animal collectives (submitted)**
