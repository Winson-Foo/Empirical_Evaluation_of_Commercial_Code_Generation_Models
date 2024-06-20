#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import os
import cv2
import time
import socket
import paramiko
import platform
import distutils
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
from enum import Enum, unique

from utils import SystemUtils
from config import resource_path

from PyInstaller.__main__ import run

""" Used to package as a single executable """

if platform.system() == 'Linux':
    if distutils.distutils_path.endswith('__init__.py'):
        distutils.distutils_path = os.path.dirname(distutils.distutils_path)

with open("./resource/VERSION", "w", encoding="utf8") as f:
    today = time.strftime("%Y%m%d", time.localtime(time.time()))
    f.write(today)


@unique
class Version(Enum):
    CPU = 'CPU'
    GPU = 'GPU'


def prepare_options(version):
    if version == Version.GPU:
        return ['tornado_server_gpu.spec', '--distpath=dist']
    else:
        return ['tornado_server.spec', '--distpath=dist']


if __name__ == '__main__':
    version = Version.CPU

    upload = False
    server_ip = ""
    username = ""
    password = ""
    model_dir = "model"
    graph_dir = "graph"

    options = prepare_options(version)
    run(options)