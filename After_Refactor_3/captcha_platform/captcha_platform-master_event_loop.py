#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import time
from watchdog.observers import Observer
from event_handler import FileEventHandler


def start_event_loop(system_config, model_path, interface_manager):
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


def main():
    # Load system configuration
    system_config = load_system_config()

    # Set model paths
    model_path = system_config["model_path"]

    # Create the interface manager
    interface_manager = InterfaceManager()  # Your implementation here

    # Start the event loop
    start_event_loop(system_config, model_path, interface_manager)


def load_system_config():
    # Your implementation here
    pass


class InterfaceManager:
    # Your implementation here
    pass


if __name__ == "__main__":
    main()