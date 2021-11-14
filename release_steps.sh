python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build
python3 -m pip install --upgrade twine

# Release idtrackerai
python3 -m build
python3 -m twine upload dist/* # this requires PyPI user and password

# Release idtrackerai-app
cd idtrackerai-app 
python3 -m build
python3 -m twine upload dist/* # this requires PyPI user and password

# Release pythonvideoannotator-module-idtrackerai
cd ../pythonvideoannotator/plugins
cd pythonvideoannotator-module-idtrackerai
python3 -m build
python3 -m twine upload dist/* # this requires PyPI user and password