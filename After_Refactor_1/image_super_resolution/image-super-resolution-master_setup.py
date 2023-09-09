from setuptools import setup, find_packages

# Define constants for package information
PACKAGE_NAME = 'ISR'
DESCRIPTION = 'Image Super Resolution'
AUTHOR = 'Francesco Cardinale'
AUTHOR_EMAIL = 'testadicardi@gmail.com'
URL = 'https://idealo.github.io/image-super-resolution/'
LICENSE = 'Apache 2.0'

# Define variables for package requirements and extras requirements
REQUIRED_PACKAGES = [
    'imageio',
    'numpy',
    'tensorflow==2.*',
    'tqdm',
    'pyaml',
    'h5py==2.10.0',
]

TEST_PACKAGES = ['pytest==4.3.0', 'pytest-cov==2.6.1']
DOC_PACKAGES = ['mkdocs==1.0.4', 'mkdocs-material==4.0.2']
GPU_PACKAGES = ['tensorflow-gpu==2.*']
DEV_PACKAGES = ['bumpversion==0.5.3']

setup(
    # Set package information using constants
    name=PACKAGE_NAME,
    version='2.2.0',
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=URL,
    license=LICENSE,

    # Define package requirements and extras requirements
    install_requires=REQUIRED_PACKAGES,
    extras_require={
        'tests': TEST_PACKAGES,
        'docs': DOC_PACKAGES,
        'gpu': GPU_PACKAGES,
        'dev': DEV_PACKAGES,
    },

    # Add classifiers to describe the package
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    
    # Use find_packages() to locate package modules
    packages=find_packages(exclude=('tests',)),
)