#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Module description: This module provides a script for packaging as a single executable.
"""

import os
import cv2
import time
import stat
import socket
import paramiko
import platform
import distutils
import tensorflow as tf
from enum import Enum, unique
from utils import SystemUtils
from config import resource_path
from PyInstaller.__main__ import run, logger

tf.compat.v1.disable_v2_behavior()

if platform.system() == 'Linux':
    if distutils.distutils_path.endswith('__init__.py'):
        distutils.distutils_path = os.path.dirname(distutils.distutils_path)

VERSION_FILE_PATH = "./resource/VERSION"
MODEL_DIR = "model"
GRAPH_DIR = "graph"


@unique
class Version(Enum):
    CPU = 'CPU'
    GPU = 'GPU'


def write_version_file(file_path):
    """
    Writes the current date as the version to the specified file.
    """
    today = time.strftime("%Y%m%d", time.localtime(time.time()))
    with open(file_path, "w", encoding="utf8") as f:
        f.write(today)


def package_as_executable(version):
    """
    Packages the script as a single executable based on the provided version.
    :param version: The version of the executable to package ('CPU' or 'GPU').
    """
    upload = False
    server_ip = ""
    username = ""
    password = ""

    if version == Version.GPU:
        opts = ['tornado_server_gpu.spec', '--distpath=dist']
    else:
        opts = ['tornado_server.spec', '--distpath=dist']
    
    run(opts)


if __name__ == '__main__':
    write_version_file(VERSION_FILE_PATH)
    package_as_executable(Version.CPU)