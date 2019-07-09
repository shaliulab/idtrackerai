# This file is part of idtracker.ai a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
# Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
# Champalimaud Foundation.
#
# idtracker.ai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (idtrackerai@gmail.com) or
# use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.
#
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P: gonzalo.polavieja@neuro.fchampalimaud.org)
import sys
from setuptools import find_packages, setup

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 6)
if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write("""
==========================
Unsupported Python version
==========================
This version of idtrackerai requires Python {}.{}, but you're trying to
install it on Python {}.{}.
""".format(*(REQUIRED_PYTHON + CURRENT_PYTHON)))
    sys.exit(1)

requirements = ['numpy >= 1.13.0',
                'natsort >= 5.0.2',
                'matplotlib >= 2.1',
                'seaborn >= 0.8',
                'tqdm >= 4.19',
                'joblib >= 0.11',
                'scikit-learn >= 0.19',
                'pyyaml >= 3.12',
                'psutil >= 5.4.3',
                'h5py >= 2.7.0',
                'xlib == 0.21',
                'msgpack-numpy >= 0.3.9',
                'msgpack-python >= 0.4.8',
                'pandas >= 0.20.2',
                'scipy >= 0.19.0',
                'sklearn >= 0.0',
                'tables >= 3.3.0',
                'dask >= 0.17.0',
                'opencv-python == 3.4.5.20',
                'tensorflow == 1.13.1',
                'confapp'
                ]


EXCLUDE_FROM_PACKAGES = [ "plots", "plots.*",
                        "test", "test.*",
                        "docs", "docs.*"]

setup(
    name='idtrackerai',
    version='3.0.2-alpha',
    python_requires='>={}.{}'.format(*REQUIRED_PYTHON),
    url='https://www.idtracker.ai/',
    author='',
    author_email='idtracker@gmail.com',
    description=('A tracking algorithm based on convolutional neural networks'),
    license='',
    packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
