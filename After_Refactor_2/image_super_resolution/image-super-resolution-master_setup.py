from setuptools import setup, find_packages

PACKAGE_NAME = "ISR"
VERSION = "2.2.0"
AUTHOR = "Francesco Cardinale"
AUTHOR_EMAIL = "testadicardi@gmail.com"
DESCRIPTION = "Image Super Resolution"
URL = "https://idealo.github.io/image-super-resolution/"
LICENSE = "Apache 2.0"

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

REQUIREMENTS = [
    "imageio",
    "numpy",
    "tensorflow==2.*",
    "tqdm",
    "pyaml",
    "h5py==2.10.0",
]

EXTRAS_REQUIRE = {
    "tests": ["pytest==4.3.0", "pytest-cov==2.6.1"],
    "docs": ["mkdocs==1.0.4", "mkdocs-material==4.0.2"],
    "gpu": ["tensorflow-gpu==2.*"],
    "dev": ["bumpversion==0.5.3"],
}

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    license=LICENSE,
    install_requires=REQUIREMENTS,
    extras_require=EXTRAS_REQUIRE,
    classifiers=CLASSIFIERS,
    packages=find_packages(exclude=("tests",)),
)