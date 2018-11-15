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
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., De Polavieja, G.G.,
# (2018). idtracker.ai: Tracking all individuals in large collectives of unmarked animals (F.R.-F. and M.G.B. contributed equally to this work. Correspondence should be addressed to G.G.d.P: gonzalo.polavieja@neuro.fchampalimaud.org)


import os
import sys
from distutils.sysconfig import get_python_lib
from setuptools import find_packages, setup
import subprocess

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

requirements = ['cython == 0.29',
                'pygame == 1.9.4',
                'numpy == 1.15.3',
                'natsort == 5.4.1',
                'seaborn == 0.9.0',
                'tqdm == 4.28.1',
                'joblib >= 0.12.5',
                'scikit-learn == 0.20.0',
                'PyAutoGUI == 0.9.38',
                'pyyaml == 3.13',
                'psutil == 5.4.8',
                'h5py == 2.8.0',
                'xlib == 0.21',
                'Kivy-Garden == 0.1.4',
                'matplotlib == 2.2.0',
                'msgpack-numpy == 0.4.4.1',
                'msgpack-python == 0.5.6',
                'natsort == 5.4.1',
                'pandas == 0.23.4',
                'PyYAML == 3.13',
                'scipy == 1.1.0',
                'sklearn >= 0.0',
                'tables == 3.4.4',
                'dask == 0.20.0']


EXCLUDE_FROM_PACKAGES = [ "plots", "plots.*",
                        "test", "test.*",
                        "docs", "docs.*"]
print(find_packages(exclude=EXCLUDE_FROM_PACKAGES))
setup(
    name='idtrackerai',
    version='2.0.0-alpha',
    python_requires='>={}.{}'.format(*REQUIRED_PYTHON),
    url='https://www.idtracker.ai/',
    author='',
    author_email='info@idtracker.ai',
    description=('A tracking algorithm based on convolutional neural networks'),
    license='',
    packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
          'console_scripts': [
              'idtrackeraiGUI = idtrackerai.gui.idtrackeraiApp:run_app',
                ]
              },
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering'
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
