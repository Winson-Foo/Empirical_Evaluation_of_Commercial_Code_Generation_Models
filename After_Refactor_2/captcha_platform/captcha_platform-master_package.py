#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import os
import time
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
from enum import Enum, unique
from config import resource_path

from PyInstaller.__main__ import run

""" Used to package as a single executable """

class Version(Enum):
    CPU = 'CPU'
    GPU = 'GPU'


if __name__ == '__main__':
    ver = Version.CPU

    upload = False
    server_ip = ""
    username = ""
    password = ""
    model_dir = "model"
    graph_dir = "graph"

    if ver == Version.GPU:
        opts = ['tornado_server_gpu.spec', '--distpath=dist']
    else:
        opts = ['tornado_server.spec', '--distpath=dist']
    run(opts)