import time
import threading
from concurrent import futures
import grpc
from compat import grpc_pb2_grpc, grpc_pb2

import optparse

from utils import ImageUtils
from interface import InterfaceManager
from config import Config
from middleware import *
from event_loop import event_loop

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class Predict(grpc_pb2_grpc.PredictServicer):

    def __init__(self, image_utils, interface_manager):
        super().__init__()
        self.image_utils = image_utils
        self.interface_manager = interface_manager

    def predict(self, request, context):
        start_time = time.time()
        bytes_batch, status = self.image_utils.get_bytes_batch(request.image)

        if self.interface_manager.total == 0:
            logger.info('There is currently no model deployment and services are not available.')
            return {"result": "", "success": False, "code": -999}

        if not bytes_batch:
            return grpc_pb2.PredictResult(result="", success=status['success'], code=status['code'])

        image_sample = bytes_batch[0]
        image_size = ImageUtils.size_of_image(image_sample)
        size_string = "{}x{}".format(image_size[0], image_size[1])
        if request.model_name:
            interface = self.interface_manager.get_by_name(request.model_name)
        else:
            interface = self.interface_manager.get_by_size(size_string)
        if not interface:
            logger.info('Service is not ready!')
            return {"result": "", "success": False, "code": 999}

        if request.need_color:
            bytes_batch = [color_extract.separate_color(_, color_map[request.need_color]) for _ in bytes_batch]

        image_batch, status = ImageUtils.get_image_batch(interface.model_conf, bytes_batch)

        if not image_batch:
            return grpc_pb2.PredictResult(result="", success=status['success'], code=status['code'])

        result = interface.predict_batch(image_batch, request.split_char)
        logger.info('[{}] - Size[{}] - Type[{}] - Site[{}] - Predict Result[{}] - {} ms'.format(
            interface.name,
            size_string,
            request.model_type,
            request.model_site,
            result,
            (time.time() - start_time) * 1000
        ))
        return grpc_pb2.PredictResult(result=result, success=status['success'], code=status['code'])


def serve(port, conf_path, model_path, graph_path):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    image_utils = ImageUtils(Config(conf_path=conf_path, model_path=model_path, graph_path=graph_path))
    interface_manager = InterfaceManager()
    threading.Thread(target=lambda: event_loop(image_utils, model_path, interface_manager)).start()
    grpc_pb2_grpc.add_PredictServicer_to_server(Predict(image_utils, interface_manager), server)
    server.add_insecure_port('[::]:' + str(port))
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-p', '--port', type="int", default=50054, dest="port")
    parser.add_option('-c', '--config', type="str", default='./config.yaml', dest="config")
    parser.add_option('-m', '--model_path', type="str", default='model', dest="model_path")
    parser.add_option('-g', '--graph_path', type="str", default='graph', dest="graph_path")
    opt, args = parser.parse_args()
    server_port = opt.port
    conf_path = opt.config
    model_path = opt.model_path
    graph_path = opt.graph_path

    logger = Config(conf_path=conf_path, model_path=model_path, graph_path=graph_path).logger
    server_host = "0.0.0.0"

    logger.info('Running on http://{}:{}/ <Press CTRL + C to quit>'.format(server_host, server_port))
    serve(server_port, conf_path, model_path, graph_path)