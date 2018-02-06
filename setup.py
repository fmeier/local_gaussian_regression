#!/usr/bin/env python
######################################################################
# Copyright (c) 2017, Max Planck Society
# \file setup.py
# \author Franziska Meier
#######################################################################
from setuptools import setup

__author__ = 'Franziska Meier'
__copyright__ = '2017, Max Planck Society'


setup(
    name='lgr',
    author='Franziska Meier',
    author_email='franzi.meier@gmail.com',
    version=1.0,
    packages=['lgr'],
    package_dir={'lgr': ''},
    install_requires=[
        'ipdb',
	'numpy',
        'jupyter',
    ],
    zip_safe=False
)
