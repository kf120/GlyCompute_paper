# -*- coding: utf-8 -*-
"""
Created on Mon May 27 07:25:25 2024

@author: kf120
"""

from setuptools import setup, find_packages

setup(
    name='glypruneabc',
    version='0.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy==1.26.4',
        'pandas==2.2.1',
        'matplotlib==3.8.4',
        'pyabc==0.12.13'
        'networkx==3.3',
        'pyvis==0.3.2',
        'glycowork==1.2.0',
        'natsort==8.4.0',
        'scipy==1.12.0'
    ],
    author='Konstantinos Flevaris',
    author_email='k.flevaris21@imperial.ac.uk',
    description='Paper on the automated analysis of glycosylation kinetics',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/kf120/GlyPruneABC_paper',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
