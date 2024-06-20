#!/usr/bin/env python

import os
import sys

from Cython.Build import cythonize
from setuptools import setup, find_packages
from setuptools.extension import Extension
from distutils.command.build_ext import build_ext

import numpy as np

def read_file(filename):
    with open(filename, 'r') as file:
        return file.read().strip()

def get_data_files(extra_dirs, base_dir='share/caiman'):
    data_files = [
        (base_dir, [
            'LICENSE.txt',
            'README.md',
            'test_demos.sh',
            'VERSION'
        ])
    ]

    for extra_dir in extra_dirs:
        extra_files = []

        for root, folders, files in os.walk(extra_dir):
            rel_path = os.path.relpath(root, extra_dir)

            for file in files:
                extra_files.append(os.path.join(rel_path, file))

        data_files.append((os.path.join(base_dir, extra_dir), extra_files))

    return data_files

def get_extension():
    if sys.platform == 'darwin':
        extra_compiler_args = ['-stdlib=libc++']  # not needed #, '-mmacosx-version-min=10.9']
    else:
        extra_compiler_args = []

    return Extension(
        "caiman.source_extraction.cnmf.oasis",
        sources=["caiman/source_extraction/cnmf/oasis.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=extra_compiler_args,
        extra_link_args=extra_compiler_args,
    )

here = os.path.abspath(os.path.dirname(__file__))

readme = read_file(os.path.join(here, 'README.md'))
version = read_file(os.path.join(here, 'VERSION'))

extra_dirs = ['bin', 'demos', 'docs', 'model']
data_files = get_data_files(extra_dirs)
binaries = ['caimanmanager.py']
data_files.append(('bin', binaries))

ext_modules = [get_extension()]

setup(
    name='caiman',
    version=version,
    author='Andrea Giovannucci, Eftychios Pnevmatikakis, Johannes Friedrich, Valentina Staneva, Ben Deverett, Erick Cobos, Jeremie Kalfon',
    author_email='pgunn@flatironinstitute.org',
    url='https://github.com/flatironinstitute/CaImAn',
    license='GPL-2',
    description='Advanced algorithms for ROI detection and deconvolution of Calcium Imaging datasets.',
    long_description=readme,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Researchers',
        'Topic :: Calcium Imaging :: Analysis Tools',
        'License :: OSI Approved :: GPL-2 License',
        'Programming Language :: Python :: 3',
    ],
    keywords='fluorescence calcium ca imaging deconvolution ROI identification',
    packages=find_packages(),
    data_files=data_files,
    install_requires=[''],
    ext_modules=cythonize(ext_modules, language_level="3"),
    cmdclass={'build_ext': build_ext}
)