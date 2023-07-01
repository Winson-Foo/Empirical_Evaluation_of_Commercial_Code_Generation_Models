#!/usr/bin/env python

import os
import sys
from setuptools import setup, find_packages
from os import path
from distutils.command.build_ext import build_ext
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

def get_data_files():
    data_files = [
        ('share/caiman', [
            'LICENSE.txt',
            'README.md',
            'test_demos.sh',
            'VERSION'
        ]),
        ('share/caiman/example_movies', [
            'example_movies/data_endoscope.tif',
            'example_movies/demoMovie.tif'
        ]),
        ('share/caiman/testdata', [
            'testdata/groundtruth.npz',
            'testdata/example.npz'
        ])
    ]

    extra_dirs = ['bin', 'demos', 'docs', 'model']
    binaries = ['caimanmanager.py']

    for part in extra_dirs:
        newpart = [
            ("share/caiman/" + d, [os.path.join(d, f) for f in files]) 
            for d, folders, files in os.walk(part)
        ]
        for newcomponent in newpart:
            data_files.append(newcomponent)

    data_files.append(['bin', binaries])
    return data_files

def setup_package():
    here = path.abspath(path.dirname(__file__))

    readme = read_file('README.md')
    version = read_file('VERSION')

    extra_compiler_args = ['-stdlib=libc++'] if sys.platform == 'darwin' else []

    ext_modules = [
        Extension(
            "caiman.source_extraction.cnmf.oasis",
            sources=["caiman/source_extraction/cnmf/oasis.pyx"],
            include_dirs=[np.get_include()],
            language="c++",
            extra_compile_args=extra_compiler_args,
            extra_link_args=extra_compiler_args
        )
    ]

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
        data_files=get_data_files(),
        install_requires=[''],
        ext_modules=cythonize(ext_modules, language_level="3"),
        cmdclass={'build_ext': build_ext}
    )

if __name__ == '__main__':
    setup_package()