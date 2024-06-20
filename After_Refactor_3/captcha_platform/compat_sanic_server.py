#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import time
import optparse
import threading
from watchdog.observers import Observer
from sanic import Sanic
from sanic.response import json

from config import Config
from utils import ImageUtils
from interface import InterfaceManager
from event_handler import FileEventHandler
from signature import Signature, ServerType
from middleware import *
from event_loop import event_loop

app = Sanic()
sign = Signature(ServerType.SANIC)
parser = optparse.OptionParser()

# Load configuration settings
conf_path = '../config.yaml'
model_path = '../model'
graph_path = '../graph'

system_config = Config(conf_path=conf_path, model_path=model_path, graph_path=graph_path)

# Set up authentication
sign.set_auth([{'accessKey': system_config.access_key, 'secretKey': system_config.secret_key}])

# Initialize logger
logger = system_config.logger

# Initialize interface manager
interface_manager = InterfaceManager()

# Start event loop in a separate thread
threading.Thread(target=lambda: event_loop(system_config, model_path, interface_manager)).start()

# Initialize image utils
image_utils = ImageUtils(system_config)


@app.route('/captcha/auth/v2', methods=['POST'])
@sign.signature_required
def auth_request(request):
    return common_request(request)


@app.route('/captcha/v1', methods=['POST'])
def no_auth_request(request):
    return common_request(request)


def common_request(request):
    """
    Common handler for captcha prediction requests.
    """
    start_time = time.time()
    
    # Check if request contains expected data
    if not request.json or 'image' not in request.json:
        logger.error('Invalid request format.')
        return

    # Check if there are any deployed models
    if interface_manager.total == 0:
        logger.info('There is currently no model deployment and services are not available.')
        return json({"message": "", "success": False, "code": -999})

    # Convert image to bytes
    bytes_batch, response = image_utils.get_bytes_batch(request.json['image'])

    if not bytes_batch:
        logger.error('Unable to convert image to bytes. Error: {}'.format(response))
        return json(response)

    # Get size of image
    image_sample = bytes_batch[0]
    image_size = ImageUtils.size_of_image(image_sample)
    size_string = "{}x{}".format(image_size[0], image_size[1])

    # Get interface based on size or model name
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
    # Parse command line arguments
    parser.add_option('-p', '--port', type="int", default=19953, dest="port")
    opt, args = parser.parse_args()
    
    server_port = opt.port
    server_host = "0.0.0.0"
    
    logger.info('Running on http://{}:{}/ <Press CTRL + C to quit>'.format(server_host, server_port))
    app.run(host=server_host, port=server_port)