"""Setup script for videoflow"""

import os.path
import setuptools
from setuptools import setup


HERE: str = os.path.abspath(os.path.dirname(__file__))
README_FILE: str = "README.md"
VERSION_FILE: str = "videoflow/version.py"
PACKAGE_NAME: str = "videoflow"
AUTHOR: str = "Jadiel de Armas"
AUTHOR_EMAIL: str = "jadielam@gmail.com"
LICENSE: str = "MIT"
URL: str = "https://github.com/videoflow/videoflow"

with open(os.path.join(HERE, README_FILE), "r") as fh:
    README: str = fh.read()

with open("requirements.txt") as f:
    REQUIRED_PACKAGES: list = f.read().splitlines()

CLASSIFIERS: list = [
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
    'Topic :: Software Development :: Libraries',
    'Topic :: Software Development :: Libraries :: Python Modules'
]

exec(open(VERSION_FILE).read())

setup(
    name = PACKAGE_NAME,
    version = __version__,
    description = "Python video stream processing library",
    long_description = README,
    long_description_content_type = "text/markdown",
    url = URL,
    author = AUTHOR,
    author_email = AUTHOR_EMAIL,
    license = LICENSE,
    packages = setuptools.find_packages(),
    include_package_data = True,
    install_requires = REQUIRED_PACKAGES,
    classifiers = CLASSIFIERS
)
