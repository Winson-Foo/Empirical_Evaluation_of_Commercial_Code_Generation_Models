from setuptools import setup
import numpy as np
from Cython.Build import cythonize


def package_description():
    text = open('README.md', 'r').read()
    startpos = text.find('## Introduction')
    return text[startpos:]


def read_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]


setup(
    name='shorttext',
    version='1.5.8',
    description="Short Text Mining",
    long_description=package_description(),
    long_description_content_type='text/markdown',
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Cython",
        "Programming Language :: C",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research"
    ],
    keywords="shorttext natural language processing text mining",
    url="https://github.com/stephenhky/PyShortTextCategorization",
    author="Kwan-Yuet Ho",
    author_email="stephenhky@yahoo.com.hk",
    license='MIT',
    ext_modules=cythonize([
        'shorttext/metrics/dynprog/dldist.pyx',
        'shorttext/metrics/dynprog/lcp.pyx'
    ]),
    packages=[
        'shorttext',
        'shorttext.utils',
        'shorttext.classifiers',
        'shorttext.classifiers.embed',
        'shorttext.classifiers.embed.nnlib',
        'shorttext.classifiers.embed.sumvec',
        'shorttext.classifiers.bow',
        'shorttext.classifiers.bow.topic',
        'shorttext.classifiers.bow.maxent',
        'shorttext.data',
        'shorttext.stack',
        'shorttext.generators',
        'shorttext.generators.bow',
        'shorttext.generators.charbase',
        'shorttext.generators.seq2seq',
        'shorttext.metrics',
        'shorttext.metrics.dynprog',
        'shorttext.metrics.wasserstein',
        'shorttext.metrics.transformers',
        'shorttext.metrics.embedfuzzy',
        'shorttext.spell'
    ],
    package_dir={'shorttext': 'shorttext'},
    package_data={
        'shorttext': [
            'data/*.csv',
            'utils/*.txt',
            'metrics/dynprog/*.pyx',
            'metrics/dynprog/*.c',
            'spell/*.pyx',
            'spell/*.c'
        ]
    },
    include_dirs=[np.get_include()],
    python_requires='>=3.7',
    setup_requires=read_requirements('setup_requirements.txt'),
    install_requires=read_requirements('requirements.txt'),
    scripts=[
        'bin/ShortTextCategorizerConsole',
        'bin/ShortTextWordEmbedSimilarity',
        'bin/WordEmbedAPI'
    ],
    test_suite="test",
    tests_requires=read_requirements('test_requirements.txt'),
    zip_safe=False
)