"""Setup script for videoflow"""

import os
import setuptools

HERE = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(HERE, "README.md")).read()

def get_version():
    exec(open('videoflow/version.py').read())
    return __version__

setuptools.setup(
    name="videoflow",
    version=get_version(),
    description="Python video stream processing library",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/videoflow/videoflow",
    author="Jadiel de Armas",
    author_email="jadielam@gmail.com",
    license="MIT",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=open(os.path.join(HERE, "requirements.txt")).readlines(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)