import os
import sys
import setuptools

from t5.version import __version__

# Constants
LONG_DESCRIPTION_FILE = 'README.md'
PACKAGE_DATA = {'': ['*.gin']}

# Get the long description from the README file.
with open(LONG_DESCRIPTION_FILE) as fp:
  long_description = fp.read()

# Setuptools setup
setuptools.setup(
    name='t5',
    version=__version__,
    description='Text-to-text transfer transformer',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Google Inc.',
    author_email='no-reply@google.com',
    url='http://github.com/google-research/text-to-text-transfer-transformer',
    license='Apache 2.0',
    packages=setuptools.find_packages(),
    package_data=PACKAGE_DATA,
    scripts=[],
    install_requires=[
        'absl-py',
        'babel',
        'editdistance',
        'immutabledict',
        'gin-config',
        'mesh-tensorflow[transformer]>=0.1.13',
        'nltk',
        'numpy',
        'pandas<2.0.0',
        'rouge-score>=0.1.2',
        'sacrebleu',
        'scikit-learn',
        'scipy',
        'sentencepiece',
        'seqio-nightly',
        'six>=1.14',
        'tfds-nightly',
        'transformers>=2.7.0',
    ],
    extras_require={
        'gcp': [
            'gevent',
            'google-api-python-client',
            'google-compute-engine',
            'google-cloud-storage',
            'oauth2client',
        ],
        'cache-tasks': ['apache-beam'],
        'test': ['pytest', 'torch'],
    },
    entry_points={
        'console_scripts': [
            't5_mesh_transformer = t5.models.mesh_transformer_main:console_entry_point',
            't5_cache_tasks = seqio.scripts.cache_tasks_main:console_entry_point',
            't5_inspect_tasks = seqio.scripts.inspect_tasks_main:console_entry_point',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='text nlp machinelearning',
)