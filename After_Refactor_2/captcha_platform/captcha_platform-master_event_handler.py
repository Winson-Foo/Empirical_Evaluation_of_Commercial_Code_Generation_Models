#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from config import ModelConfig, Config
from graph_session import GraphSession
from utils import PathUtils


class FileEventHandler(FileSystemEventHandler):
    def __init__(self, conf: Config, model_conf_path: str, interface_manager: 'InterfaceManager'):
        super().__init__()
        self.conf = conf
        self.logger = self.conf.logger
        self.name_map = {}
        self.model_conf_path = model_conf_path
        self.interface_manager = interface_manager
        self.init()

    def init(self):
        model_list = os.listdir(self.model_conf_path)
        model_list = [os.path.join(self.model_conf_path, i) for i in model_list if i.endswith("yaml")]
        for model in model_list:
            self._add(model, is_first=True)
        if self.interface_manager.total == 0:
            self.logger.info(
                f"\n - Number of interfaces: {self.interface_manager.total}"
                f"\n - There is currently no model deployment"
                f"\n - Services are not available"
                f"\n[ Please check the graph and model path whether the pb file and yaml file are placed. ]"
            )
        else:
            self.logger.info(
                f"\n - Number of interfaces: {self.interface_manager.total}"
                f"\n - Current online interface:"
                f"\n\t - " + "\n\t - ".join([f"[{v}]" for k, v in self.name_map.items()])
                f"\n - The default Interface is: {self.interface_manager.default_name}"
            )

    def _add(self, src_path, is_first=False, count=0):
        try:
            model_path = str(src_path)
            path_exists = os.path.exists(model_path)
            if not path_exists and count > 0:
                self.logger.error(f"{model_path} not found, retry attempt is terminated.")
                return
            if 'model_demo.yaml' in model_path:
                self.logger.warning(
                    "\n-------------------------------------------------------------------\n"
                    "- Found that the model_demo.yaml file exists, \n"
                    "- the loading is automatically ignored. \n"
                    "- If it is used for the first time, \n"
                    "- please copy it as a template. \n"
                    "- and do not use the reserved character \"model_demo.yaml\" as the file name."
                    "\n-------------------------------------------------------------------"
                )
                return
            if model_path.endswith("yaml"):
                model_conf = ModelConfig(self.conf, model_path)
                inner_name = model_conf.model_name
                inner_size = model_conf.size_string
                inner_key = PathUtils.get_file_name(model_path)
                for k, v in self.name_map.items():
                    if inner_size in v:
                        self.logger.warning(
                            "\n-------------------------------------------------------------------\n"
                            "- The current model {inner_key} is the same size [{inner_size}] as the loaded model {k}. \n"
                            "- Only one of the smart calls can be called. \n"
                            "- If you want to refer to one of them, \n"
                            "- please use the model key or model type to find it."
                            "\n-------------------------------------------------------------------"
                        )
                        break

                inner_value = model_conf.model_name
                graph_session = GraphSession(model_conf)
                if graph_session.loaded:
                    interface = Interface(graph_session)
                    if inner_name == self.conf.default_model:
                        self.interface_manager.set_default(interface)
                    else:
                        self.interface_manager.add(interface)
                    self.logger.info(f"{'Inited' if is_first else 'Added'} a new model: {inner_value} ({inner_key})")
                    self.name_map[inner_key] = inner_value
                    if src_path in self.interface_manager.invalid_group:
                        self.interface_manager.invalid_group.pop(src_path)
                else:
                    self.interface_manager.report(src_path)
                    if count < 12 and not is_first:
                        time.sleep(5)
                        return self._add(src_path, is_first=is_first, count=count + 1)

        except Exception as e:
            self.interface_manager.report(src_path)
            self.logger.error(e)

    def delete(self, src_path):
        try:
            model_path = str(src_path)
            if model_path.endswith("yaml"):
                inner_key = PathUtils.get_file_name(model_path)
                graph_name = self.name_map.get(inner_key)
                self.interface_manager.remove_by_name(graph_name)
                self.name_map.pop(inner_key)
                self.logger.info(f"Unload the model: {graph_name} ({inner_key})")
        except Exception as e:
            self.logger.error(f"Config File {str(e).replace("'", '')} does not exist.")

    def on_created(self, event):
        if event.is_directory:
            self.logger.info(f"directory created:{event.src_path}")
        else:
            model_path = str(event.src_path)
            self._add(model_path)
            self.logger.info(
                f"\n - Number of interfaces: {len(self.interface_manager.group)}"
                f"\n - Current online interface:"
                f"\n\t - " + "\n\t - ".join([f"[{v}]" for k, v in self.name_map.items()])
                f"\n - The default Interface is: {self.interface_manager.default_name}"
            )

    def on_deleted(self, event):
        if event.is_directory:
            self.logger.info(f"directory deleted:{event.src_path}")
        else:
            model_path = str(event.src_path)
            if model_path in self.interface_manager.invalid_group:
                self.interface_manager.invalid_group.pop(model_path)
            inner_key = PathUtils.get_file_name(model_path)
            if inner_key in self.name_map:
                self.delete(model_path)
            self.logger.info(
                f"\n - Number of interfaces: {len(self.interface_manager.group)}"
                f"\n - Current online interface:"
                f"\n\t - " + "\n\t - ".join([f"[{v}]" for k, v in self.name_map.items()])
                f"\n - The default Interface is: {self.interface_manager.default_name}"
            )


class InterfaceManager:
    def __init__(self):
        self.group = []
        self.default_interface = None
        self.invalid_group = {}

    def add(self, interface: 'Interface'):
        self.group.append(interface)

    def set_default(self, interface: 'Interface'):
        self.default_interface = interface

    def remove_by_name(self, name: str):
        interfaces_to_remove = [interface for interface in self.group if interface.name == name]
        for interface in interfaces_to_remove:
            self.group.remove(interface)

    @property
    def total(self):
        return len(self.group)


if __name__ == "__main__":
    model_conf_path = ""  # Add the actual model configuration path here
    conf = Config()  # Add the necessary configuration parameters
    interface_manager = InterfaceManager()
    event_handler = FileEventHandler(conf, model_conf_path, interface_manager)
    observer = Observer()
    observer.schedule(event_handler, event_handler.model_conf_path, True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()