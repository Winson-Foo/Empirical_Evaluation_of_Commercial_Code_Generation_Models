import os
from setuptools import find_packages, setup
from nflows.version import VERSION

# Helper function to read the README file
def read_file(filename):
    with open(filename, encoding='utf-8') as f:
        return f.read()

# Set up the package
setup(
    name="nflows",
    version=VERSION,
    description="Normalizing flows in PyTorch.",
    long_description=read_file(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md')),
    long_description_content_type='text/markdown',
    url="https://github.com/bayesiains/nflows",
    download_url='https://github.com/bayesiains/nflows/archive/v0.14.tar.gz',
    author="Conor Durkan, Artur Bekasov, George Papamakarios, Iain Murray",
    packages=find_packages(exclude=["tests"]),
    license="MIT",
    install_requires=[
        "matplotlib",
        "numpy",
        "tensorboard",
        "torch",
        "tqdm",
        "umnn"
    ],
    dependency_links=[],
)