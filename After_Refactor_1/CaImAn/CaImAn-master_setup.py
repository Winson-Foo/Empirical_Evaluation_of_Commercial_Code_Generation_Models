#!/usr/bin/env python

import os
import sys
from setuptools import setup, find_packages
from setuptools.extension import Extension
from os import path
import numpy as np
from Cython.Build import cythonize
from distutils.command.build_ext import build_ext

def read_file(file_path):
    """
    Read the contents of a file and return the content as a string.

    Args:
        file_path (str): Path to the file to be read.

    Returns:
        str: File content.
    """
    with open(file_path, 'r') as file:
        return file.read()

def get_data_files():
    """
    Generate data_files list based on the binary and extra_dirs.

    Returns:
        list: List of data files.
    """
    binaries = ['caimanmanager.py']
    extra_dirs = ['bin', 'demos', 'docs', 'model']
    data_files = [('share/caiman', [
        'LICENSE.txt',
        'README.md',
        'test_demos.sh',
        'VERSION'
    ])]

    for part in extra_dirs:
        newpart = [
            ("share/caiman/" + d, [os.path.join(d,f) for f in files])
            for d, folders, files in os.walk(part)
        ]
        for newcomponent in newpart:
            data_files.append(newcomponent)

    data_files.append(['bin', binaries])
    return data_files

def get_compiler_args():
    """
    Get extra compiler arguments based on the platform.

    Returns:
        list: List of extra compiler arguments.
    """
    if sys.platform == 'darwin':
        # see https://github.com/pandas-dev/pandas/issues/23424
        return ['-stdlib=libc++']  # not needed #, '-mmacosx-version-min=10.9']
    else:
        return []

ext_modules = [Extension(
    "caiman.source_extraction.cnmf.oasis",
    sources=["caiman/source_extraction/cnmf/oasis.pyx"],
    include_dirs=[np.get_include()],
    language="c++",
    extra_compile_args=get_compiler_args(),
    extra_link_args=get_compiler_args(),
)]

setup(
    name='caiman',
    version=read_file('VERSION').strip(),
    author='Andrea Giovannucci, Eftychios Pnevmatikakis, Johannes Friedrich, Valentina Staneva, Ben Deverett, Erick Cobos, Jeremie Kalfon',
    author_email='pgunn@flatironinstitute.org',
    url='https://github.com/flatironinstitute/CaImAn',
    license='GPL-2',
    description='Advanced algorithms for ROI detection and deconvolution of Calcium Imaging datasets.',
    long_description=read_file('README.md'),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Researchers',
        'Topic :: Calcium Imaging :: Analysis Tools',
        'License :: OSI Approved :: GPL-2 License',
        'Programming Language :: Python :: 3',
    ],
    keywords='fluorescence calcium ca imaging deconvolution ROI identification',
    packages=find_packages(),
    data_files=get_data_files(),
    install_requires=[''],
    ext_modules=cythonize(ext_modules, language_level="3"),
    cmdclass={'build_ext': build_ext}
)