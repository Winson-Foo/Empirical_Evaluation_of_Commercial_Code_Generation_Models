import os.path
from setuptools import find_packages, setup

HERE = os.path.abspath(os.path.dirname(__file__))
README_PATH = os.path.join(HERE, "README.md")
VERSION_PATH = os.path.join(HERE, "videoflow", "version.py")

def read_file(path):
    with open(path) as file:
        return file.read()

def read_version():
    global_vars = {}
    exec(read_file(VERSION_PATH), global_vars)
    return global_vars['__version__']

NAME = "videoflow"
DESCRIPTION = "Python video stream processing library"
LICENSE = "MIT"
AUTHOR = "Jadiel de Armas"
AUTHOR_EMAIL = "jadielam@gmail.com"
URL = "https://github.com/videoflow/videoflow"
CLASSIFIERS = [
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
    'Topic :: Software Development :: Libraries',
    'Topic :: Software Development :: Libraries :: Python Modules'
]

REQUIRED_PACKAGES = [
    'numpy>=1.9.1',
    'opencv-python>=4.0.0',
    'six>=1.9.0',
    'requests>=2.22.0'
]

def setup_package():
    setup(
        name=NAME,
        version=read_version(),
        description=DESCRIPTION,
        long_description=read_file(README_PATH),
        long_description_content_type="text/markdown",
        url=URL,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        license=LICENSE,
        packages=find_packages(),
        include_package_data=True,
        install_requires=REQUIRED_PACKAGES,
        classifiers=CLASSIFIERS
    )

if __name__ == '__main__':
    setup_package()