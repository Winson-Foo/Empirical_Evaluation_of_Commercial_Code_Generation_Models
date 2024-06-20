#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import time
from watchdog.observers import Observer
from event_handler import FileEventHandler

def start_event_observer(system_config, model_path, interface_manager):
    observer = Observer()
    event_handler = FileEventHandler(system_config, model_path, interface_manager)
    observer.schedule(event_handler, event_handler.model_conf_path, True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    system_config = "path/to/system/config"
    model_path = "path/to/model"
    interface_manager = "path/to/interface/manager"
    start_event_observer(system_config, model_path, interface_manager)