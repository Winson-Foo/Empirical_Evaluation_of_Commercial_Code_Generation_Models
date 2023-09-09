from setuptools import setup
import numpy as np
from Cython.Build import cythonize


README_FILE = 'README.md'
REQUIREMENTS_FILE = 'requirements.txt'
SETUP_REQUIREMENTS_FILE = 'setup_requirements.txt'
TEST_REQUIREMENTS_FILE = 'test_requirements.txt'
PACKAGES = [
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
]
PACKAGE_DIR = {'shorttext': 'shorttext'}
PACKAGE_DATA = {
    'shorttext': [
        'data/*.csv',
        'utils/*.txt',
        'metrics/dynprog/*.pyx',
        'metrics/dynprog/*.c',
        'spell/*.pyx',
        'spell/*.c'
    ]
}


def package_description() -> str:
    """
    Returns the package description from README.md file.
    """
    with open(README_FILE, 'r') as f:
        text = f.read()
    startpos = text.find('## Introduction')
    return text[startpos:]


def read_requirements(file_name: str) -> list[str]:
    """
    Reads the requirements from the specified file and returns as a list.
    """
    with open(file_name, 'r') as f:
        return [package_string.strip() for package_string in f.readlines()]


def setup_shorttext() -> None:
    """
    Setup function.
    """
    ext_modules = cythonize([
        'shorttext/metrics/dynprog/dldist.pyx',
        'shorttext/metrics/dynprog/lcp.pyx'
    ])

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
        ext_modules=ext_modules,
        packages=PACKAGES,
        package_dir=PACKAGE_DIR,
        package_data=PACKAGE_DATA,
        include_dirs=[np.get_include()],
        python_requires='>=3.7',
        setup_requires=read_requirements(SETUP_REQUIREMENTS_FILE),
        install_requires=read_requirements(REQUIREMENTS_FILE),
        scripts=[
            'bin/ShortTextCategorizerConsole',
            'bin/ShortTextWordEmbedSimilarity',
            'bin/WordEmbedAPI'
        ],
        test_suite="test",
        tests_requires=read_requirements(TEST_REQUIREMENTS_FILE),
        zip_safe=False
    )


if __name__ == '__main__':
    setup_shorttext()