#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import time
import optparse
import threading
from config import Config
from utils import ImageUtils
from interface import InterfaceManager
from watchdog.observers import Observer
from event_handler import FileEventHandler
from sanic import Sanic
from sanic.response import json
from signature import Signature, ServerType
from middleware import *
from event_loop import event_loop

app = Sanic()
sign = Signature(ServerType.SANIC)
parser = optparse.OptionParser()

conf_path = '../config.yaml'
model_path = '../model'
graph_path = '../graph'

system_config = Config(conf_path=conf_path, model_path=model_path, graph_path=graph_path)
sign.set_auth([{'accessKey': system_config.access_key, 'secretKey': system_config.secret_key}])
logger = system_config.logger
interface_manager = InterfaceManager()
threading.Thread(target=lambda: event_loop(system_config, model_path, interface_manager)).start()

image_utils = ImageUtils(system_config)


@app.route('/captcha/auth/v2', methods=['POST'])
@sign.signature_required  # This decorator is required for certification.
def auth_request(request):
    return common_request(request)


@app.route('/captcha/v1', methods=['POST'])
def no_auth_request(request):
    return common_request(request)


def common_request(request):
    """
    This api is used for captcha prediction without authentication
    :return:
    """
    start_time = time.time()
    if not request.json or 'image' not in request.json:
        print(request.json)
        return

    if interface_manager.total == 0:
        logger.info('There is currently no model deployment and services are not available.')
        return json({"message": "", "success": False, "code": -999})

    bytes_batch, response = image_utils.get_bytes_batch(request.json['image'])

    if not bytes_batch:
        logger.error('Type[{}] - Site[{}] - Response[{}] - {} ms'.format(
            request.json['model_type'], request.json['model_site'], response,
            (time.time() - start_time) * 1000)
        )
        return json(response)

    image_sample = bytes_batch[0]
    image_size = ImageUtils.size_of_image(image_sample)
    size_string = "{}x{}".format(image_size[0], image_size[1])

    if 'model_name' in request.json:
        interface = interface_manager.get_by_name(request.json['model_name'])
    else:
        interface = interface_manager.get_by_size(size_string)

    split_char = request.json['split_char'] if 'split_char' in request.json else interface.model_conf.split_char

    if 'need_color' in request.json and request.json['need_color']:
        bytes_batch = [color_extract.separate_color(_, color_map[request.json['need_color']]) for _ in bytes_batch]

    image_batch, response = ImageUtils.get_image_batch(interface.model_conf, bytes_batch)

    if not image_batch:
        logger.error('[{}] - Size[{}] - Name[{}] - Response[{}] - {} ms'.format(
            interface.name, size_string, request.json.get('model_name'), response,
            (time.time() - start_time) * 1000)
        )
        return json(response)

    result = interface.predict_batch(image_batch, split_char)
    logger.info('[{}] - Size[{}] - Predict Result[{}] - {} ms'.format(
        interface.name,
        size_string,
        result,
        (time.time() - start_time) * 1000
    ))
    response['message'] = result
    return json(response)


if __name__ == "__main__":

    parser.add_option('-p', '--port', type="int", default=19953, dest="port")

    opt, args = parser.parse_args()
    server_port = opt.port



    server_host = "0.0.0.0"

    logger.info('Running on http://{}:{}/ <Press CTRL + C to quit>'.format(server_host, server_port))
    app.run(host=server_host, port=server_port)
