import os
import time
from watchdog.events import FileSystemEventHandler

from config import ModelConfig, Config
from graph_session import GraphSession
from interface import InterfaceManager, Interface
from utils import PathUtils

class FileEventHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.conf = Config()
        self.logger = self.conf.logger
        self.name_map = {}
        self.model_conf_path = ""
        self.interface_manager = InterfaceManager()
        self.init()

    def init(self):
        model_list = os.listdir(self.model_conf_path)
        model_list = [os.path.join(self.model_conf_path, i) for i in model_list if i.endswith("yaml")]
        
        for model in model_list:
            self._add(model, is_first=True)

        if self.interface_manager.total == 0:
            self.logger.info("Number of interfaces: 0")
            self.logger.info("There is currently no model deployment")
            self.logger.info("Services are not available")
            self.logger.info("[Please check the graph and model path whether the pb file and yaml file are placed.]")
        else:
            self.logger.info("Number of interfaces: {}".format(self.interface_manager.total))
            self.logger.info("Current online interfaces: ")
            self.logger.info("\t" + "\n\t".join(["[{}]".format(v) for k, v in self.name_map.items()]))
            self.logger.info("The default interface is: {}".format(self.interface_manager.default_name))

    def _add(self, src_path, is_first=False, count=0):
        try:
            model_path = str(src_path)
            path_exists = os.path.exists(model_path)
            
            if not path_exists and count > 0:
                self.logger.error("{} not found, retry attempt is terminated.".format(model_path))
                return

            if 'model_demo.yaml' in model_path:
                self.logger.warning("Found that the model_demo.yaml file exists, the loading is automatically ignored.")
                self.logger.warning("If it is used for the first time, please copy it as a template.")
                self.logger.warning("And do not use the reserved character \"model_demo.yaml\" as the file name.")
                return

            if model_path.endswith("yaml"):
                model_conf = ModelConfig(self.conf, model_path)
                inner_name = model_conf.model_name
                inner_size = model_conf.size_string
                inner_key = PathUtils.get_file_name(model_path)

                for k, v in self.name_map.items():
                    if inner_size in v:
                        self.logger.warning(
                            "The current model {} is the same size [{}] as the loaded model {}.".format(
                                inner_key, inner_size, k
                            )
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

                    self.logger.info("{} a new model: {} ({})".format(
                        "Inited" if is_first else "Added", inner_value, inner_key
                    ))

                    self.name_map[inner_key] = inner_value

                    if src_path in self.interface_manager.invalid_group:
                        self.interface_manager.invalid_group.pop(src_path)
                else:
                    self.interface_manager.report(src_path)

                    if count < 12 and not is_first:
                        time.sleep(5)
                        return self._add(src_path, is_first=is_first, count=count+1)

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
                self.logger.info("Unload the model: {} ({})".format(graph_name, inner_key))
        except Exception as e:
            self.logger.error("Config File [{}] does not exist.".format(str(e).replace("'", "")))

    def on_created(self, event):
        if event.is_directory:
            self.logger.info("Directory created: {}".format(event.src_path))
        else:
            model_path = str(event.src_path)
            self._add(model_path)
            self.logger.info("Number of interfaces: {}".format(len(self.interface_manager.group)))
            self.logger.info("Current online interfaces: ")
            self.logger.info("\t" + "\n\t".join(["[{}]".format(v) for k, v in self.name_map.items()]))
            self.logger.info("The default interface is: {}".format(self.interface_manager.default_name))

    def on_deleted(self, event):
        if event.is_directory:
            self.logger.info("Directory deleted: {}".format(event.src_path))
        else:
            model_path = str(event.src_path)
            if model_path in self.interface_manager.invalid_group:
                self.interface_manager.invalid_group.pop(model_path)
            inner_key = PathUtils.get_file_name(model_path)
            if inner_key in self.name_map:
                self.delete(model_path)
            
            self.logger.info("Number of interfaces: {}".format(len(self.interface_manager.group)))
            self.logger.info("Current online interfaces: ")
            self.logger.info("\t" + "\n\t".join(["[{}]".format(v) for k, v in self.name_map.items()]))
            self.logger.info("The default interface is: {}".format(self.interface_manager.default_name))