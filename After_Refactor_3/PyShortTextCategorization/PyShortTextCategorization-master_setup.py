import numpy as np
from setuptools import setup
from Cython.Build import cythonize

README_FILE = 'README.md'
REQUIREMENTS_FILES = ['requirements.txt', 'setup_requirements.txt', 'test_requirements.txt']
PACKAGE_NAME = 'shorttext'
VERSION_NUMBER = '1.5.8'


def package_description() -> str:
    """
    Extracts the package description from the README file.
    """
    with open(README_FILE, 'r') as f:
        text = f.read()
        startpos = text.find('## Introduction')
        return text[startpos:]

        
def load_requirements(req_file: str) -> list:
    """
    Loads a list of package names from a requirements file.
    """
    with open(req_file, 'r') as f:
        return [pkg_name.strip() for pkg_name in f]

        
def setup_package() -> None:
    """
    Sets up the package and its dependencies.
    """
    ext_modules = cythonize([
        'shorttext/metrics/dynprog/dldist.pyx',
        'shorttext/metrics/dynprog/lcp.pyx'])

    classifiers = [
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
    ]
    
    packages = [
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
        'shorttext.spell',
    ]
    
    package_dir = {
        'shorttext': 'shorttext'
    }

    package_data = {
        'shorttext': [
            'data/*.csv',
            'utils/*.txt',
            'metrics/dynprog/*.pyx',
            'metrics/dynprog/*.c',
            'spell/*.pyx',
            'spell/*.c'
        ]
    }

    include_dirs = [
        np.get_include()
    ]
    
    scripts = [
        'bin/ShortTextCategorizerConsole',
        'bin/ShortTextWordEmbedSimilarity',
        'bin/WordEmbedAPI'
    ]
    
    setup(
        name=PACKAGE_NAME,
        version=VERSION_NUMBER,
        description="Short Text Mining",
        long_description=package_description(),
        long_description_content_type='text/markdown',
        classifiers=classifiers,
        keywords="shorttext natural language processing text mining",
        url="https://github.com/stephenhky/PyShortTextCategorization",
        author="Kwan-Yuet Ho",
        author_email="stephenhky@yahoo.com.hk",
        license='MIT',
        ext_modules=ext_modules,
        packages=packages,
        package_dir=package_dir,
        package_data=package_data,
        include_dirs=include_dirs,
        python_requires='>=3.7',
        setup_requires=load_requirements(REQUIREMENTS_FILES[1]),
        install_requires=load_requirements(REQUIREMENTS_FILES[0]),
        tests_requires=load_requirements(REQUIREMENTS_FILES[2]),
        scripts=scripts,
        test_suite="test",
        zip_safe=False
    )

if __name__ == '__main__':
    setup_package()