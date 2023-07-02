#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
# Gunicorn deploy file.

import multiprocessing
from config import *

bind = Config.bind
workers = Config.get_num_workers()
backlog = Config.backlog
debug = Config.debug
daemon = Config.daemon
proc_name = Config.proc_name
pidfile = Config.pidfile
errorlog = Config.errorlog
accesslog = Config.accesslog
loglevel = Config.loglevel
timeout = Config.timeout