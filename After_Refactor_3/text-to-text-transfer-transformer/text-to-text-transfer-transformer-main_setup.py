# Copyright 2023 The T5 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Install T5."""

import os
import sys
import setuptools

from t5.version import __version__

def get_long_description():
    """Get the long description from the README file."""
    with open('README.md') as fp:
        return fp.read()

def setup_package():
    """Set up the T5 package."""
    version_path = os.path.join(os.path.dirname(__file__), 't5')
    sys.path.append(version_path)

    setuptools.setup(
        name='t5',
        version=__version__,
        description='Text-to-text transfer transformer',
        long_description=get_long_description(),
        long_description_content_type='text/markdown',
        author='Google Inc.',
        author_email='no-reply@google.com',
        url='http://github.com/google-research/text-to-text-transfer-transformer',
        license='Apache 2.0',
        packages=setuptools.find_packages(),
        package_data={
            '': ['*.gin'],
        },
        install_requires=get_install_requires(),
        extras_require={
            'gcp': get_gcp_requirements(),
            'cache-tasks': ['apache-beam'],
            'test': ['pytest', 'torch'],
        },
        entry_points=get_entry_points(),
        classifiers=get_classifiers(),
        keywords='text nlp machinelearning',
    )

def get_install_requires():
    """Get the required packages for installation."""
    return [
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
    ]

def get_gcp_requirements():
    """Get the required packages for GCP."""
    return [
        'gevent',
        'google-api-python-client',
        'google-compute-engine',
        'google-cloud-storage',
        'oauth2client',
    ]

def get_entry_points():
    """Get the entry points for the console scripts."""
    return {
        'console_scripts': [
            't5_mesh_transformer = t5.models.mesh_transformer_main:console_entry_point',
            't5_cache_tasks = seqio.scripts.cache_tasks_main:console_entry_point',
            't5_inspect_tasks = seqio.scripts.inspect_tasks_main:console_entry_point',
        ],
    }

def get_classifiers():
    """Get the classifiers for the package."""
    return [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]

if __name__ == '__main__':
    setup_package()