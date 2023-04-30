"""Setup script for videoflow"""

import os.path
import setuptools
from setuptools import setup

# The directory containing this file
HERE = os.path.abspath(os.path.dirname(__file__))

# The text of the README file
with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()

__version__ = None  # set __version__ in this exec() call
exec(open('videoflow/version.py').read())
# This call to setup() does all the work
setup(
    name = "videoflow",
    version = __version__,
    description="Python video stream processing library",
    long_description = README,
    long_description_content_type = "text/markdown",
    url = "https://github.com/videoflow/videoflow",
    author = "Jadiel de Armas",
    author_email = "jadielam@gmail.com",
    license = "MIT",
    packages = setuptools.find_packages(),
    include_package_data = True,
    install_requires = [
        'numpy>=1.9.1',
        'opencv-python>=4.0.0',
        'six>=1.9.0',
        'requests>=2.22.0'
    ],
    classifiers = [
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)
