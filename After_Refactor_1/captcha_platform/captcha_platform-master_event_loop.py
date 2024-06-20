#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import time
from watchdog.observers import Observer
from event_handler import FileEventHandler


def start_event_loop(system_config: dict, model_path: str, interface_manager: dict) -> None:
    """
    Starts the event loop to monitor changes in the model configuration file.
    :param system_config: The system configuration.
    :param model_path: The path to the model.
    :param interface_manager: The interface manager.
    :return: None
    """
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
    # Load system configuration
    system_config = {}

    # Load model path
    model_path = ""

    # Load interface manager
    interface_manager = {}

    start_event_loop(system_config, model_path, interface_manager)