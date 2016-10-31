# -*- coding: utf-8 -*-
import sys

VERSION = '0.0.1'
DESCRIPTION = 'High-level API for Echo State Network'

from setuptools import setup
setup(
    name='esnpy',
    version=VERSION,
    author='NiZiL',
    author_email='biasutto.t@gmail.com',

    url='https://github.com/NiZiL/esnpy',
    download_url='https://github.com/NiZiL/esnpy/tarball/'+VERSION,

    description=DESCRIPTION,
    keywords=['machine learning', 'neural network', 'echo state network', 'reservoir computing'],

    packages=['esnpy'],

    install_requires=['numpy', 'scipy', 'scikit-learn']
)
