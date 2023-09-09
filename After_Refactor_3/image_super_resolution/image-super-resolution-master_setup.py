from setuptools import find_packages, setup

DEPENDENCIES = [
    'h5py==2.10.0',
    'imageio',
    'numpy',
    'pyaml',
    'tensorflow==2.*',
    'tqdm',
]

EXTRAS_REQUIRE = {
    'dev': ['bumpversion==0.5.3'],
    'docs': ['mkdocs==1.0.4', 'mkdocs-material==4.0.2'],
    'gpu': ['tensorflow-gpu==2.*'],
    'tests': ['pytest==4.3.0', 'pytest-cov==2.6.1'],
}

AUTHOR = 'Francesco Cardinale'
AUTHOR_EMAIL = 'testadicardi@gmail.com'
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Topic :: Software Development :: Libraries',
    'Topic :: Software Development :: Libraries :: Python Modules',
]
DESCRIPTION = 'Image Super Resolution'
LICENSE = 'Apache 2.0'

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

NAME = 'ISR'
PACKAGES = find_packages(exclude=('tests',))
VERSION = '2.2.0'

setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    license=LICENSE,
    install_requires=DEPENDENCIES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=CLASSIFIERS,
    packages=PACKAGES,
)