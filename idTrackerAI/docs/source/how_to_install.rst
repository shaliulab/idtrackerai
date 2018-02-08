Installation
============

^^^^^^^^^^^^^^^^^^^^^^^
Debianoid Linux and Mac
^^^^^^^^^^^^^^^^^^^^^^^

* Python2.7 should be installed.
* Install idtrackerai by typing the following commands to the terminal::

     sudo pip install idtrackerai

* Do not close the terminal, you are going to need it.

^^^^^^^^^
Windows 7
^^^^^^^^^

* Install `Python 2.7 <http://www.python.org/ftp/python/2.7/python-2.7.msi>`_
* Install `Python Setuptools <http://pypi.python.org/packages/2.7/s/setuptools/setuptools-0.6c11.win32-py2.7.exe#md5=57e1e64f6b7c7f1d2eddfc9746bbaf20>`_ (a package manager)
* Set PATH environment variable for Python scripts:

  - Right-click *Computer*
  - Click *Properties*
  - Go to the *Advanced system settings* tab
  - Click the *Environment Variables* button
  - From *System Variables*, select *Path*, and click *Edit*
  - Assuming you installed Python to ``C:\Python27`` (the default), add this to the end of *Variable value*::

       C:\Python27;C:\Python27\Scripts

* Launch the terminal: Click *Start*, find *Powershell*, click it.
* Install idtrackerai by typing the following commands to the terminal::

     easy_install pip
     pip install idtrackerai

* Do not close the terminal, you are going to need it.
