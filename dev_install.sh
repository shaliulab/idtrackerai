pip install -e .[dev]
cd idtrackerai-app
pip install -e .
cd pyforms-gui
pip install -e .
cd ../pyforms-terminal
pip install -e .
cd ../../python-video-annotator
python utils/install.py
cd plugins/pythonvideoannotator-module-idtrackerai/
pip install -e .