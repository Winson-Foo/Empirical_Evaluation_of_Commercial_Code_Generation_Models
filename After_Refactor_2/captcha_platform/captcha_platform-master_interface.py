#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import os
import time
from typing import List

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class GraphSession:

    def __init__(self, model_conf):
        self.model_conf = model_conf
        self.graph_name = self.model_conf.graph_name
        self.version = self.model_conf.version
        self.loaded = False
        self.session = None

    def load(self):
        # Load the graph session
        pass

    def destroy(self):
        # Destroy the graph session
        pass


class Interface:

    def __init__(self, graph_session: GraphSession):
        self.graph_session = graph_session
        self.model_conf = graph_session.model_conf
        self.size_str = self.model_conf.size_string
        self.graph_name = self.graph_session.graph_name
        self.version = self.graph_session.version
        self.model_category = self.model_conf.category_param
        self.sess = self.graph_session.session
        self.dense_decoded = self.sess.graph.get_tensor_by_name("dense_decoded:0")
        self.x = self.sess.graph.get_tensor_by_name('input:0')

    @property
    def name(self) -> str:
        return self.graph_name

    @property
    def size(self) -> str:
        return self.size_str

    def predict_batch(self, image_batch: List[str], output_split: str = None) -> str:
        # Perform batch prediction
        pass


class InterfaceManager:

    def __init__(self, default_interface: Interface = None):
        self.interfaces = []
        self.invalid_interfaces = {}
        self.set_default(default_interface)

    def add(self, interface: Interface):
        if interface not in self.interfaces:
            self.interfaces.append(interface)

    def remove(self, interface: Interface):
        if interface in self.interfaces:
            interface.destroy()
            self.interfaces.remove(interface)

    def report_invalid_interface(self, model_name: str):
        self.invalid_interfaces[model_name] = {"create_time": time.asctime(time.localtime(time.time()))}

    def remove_by_name(self, graph_name: str):
        interface = self.get_by_name(graph_name)
        self.remove(interface)

    def get_by_size(self, size: str, return_default: bool = True) -> Interface:
        # Return the interface with matching size
        pass

    def get_by_name(self, key: str, return_default: bool = True) -> Interface:
        # Return the interface with matching name
        pass

    @property
    def default(self) -> Interface:
        return self.interfaces[0] if self.interfaces else None

    @property
    def default_name(self) -> str:
        default_interface = self.default
        return default_interface.graph_name if default_interface else None

    @property
    def total(self) -> int:
        return len(self.interfaces)

    @property
    def online_names(self) -> List[str]:
        return [interface.name for interface in self.interfaces]

    def set_default(self, default_interface: Interface):
        if default_interface:
            self.interfaces.insert(0, default_interface)