#!/usr/bin/env python

from distutils.core import setup

setup(name='Magic Mirror',
      version='1.0',
      description='Magic Mirror that turns on when you smile',
      author='Arijit Banerjee',
      author_email='Arijit.Banerjee@gmail.com',
      install_requires=[
          'opencv-python==4.7.0.68',
          'numpy==1.21.6',
          'imutils==0.5.4',
          'h5py<3.0.0',
          "RPi.GPIO==0.7.1; os_name=='linux'",
          "opencv-python-headless; os_name=='posix'"
      ],
      scripts=['magicmirror']
   )