#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import time
import optparse
import threading
from flask import Flask, request, jsonify, abort
from flask_caching import Cache
from gevent.pywsgi import WSGIServer
from geventwebsocket.handler import WebSocketHandler
from watchdog.observers import Observer

from config import Config
from utils import ImageUtils
from constants import Response
from interface import InterfaceManager
from signature import Signature, ServerType
from event_handler import FileEventHandler
from middleware import *

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

system_config = Config(conf_path='../config.yaml', model_path='../model', graph_path='../graph')
sign = Signature(ServerType.FLASK, system_config)
_except = Response(system_config.response_def_map)
route_map = {i['Class']: i['Route'] for i in system_config.route_map}

sign.set_auth([{'accessKey': system_config.access_key, 'secretKey': system_config.secret_key}])
logger = system_config.logger
interface_manager = InterfaceManager()
image_utils = ImageUtils(system_config)

# The order cannot be changed, it must be before the flask.
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.errorhandler(400)
def bad_request(error=None):
    message = "Bad Request"
    return jsonify(message=message, code=error.code, success=False)

@app.errorhandler(500)
def internal_server_error(error=None):
    message = 'Internal Server Error'
    return jsonify(message=message, code=500, success=False)

@app.errorhandler(404)
def not_found_error(error=None):
    message = '404 Not Found'
    return jsonify(message=message, code=error.code, success=False)

@app.errorhandler(403)
def permission_denied(error=None):
    message = 'Forbidden'
    return jsonify(message=message, code=error.code, success=False)

@app.route(route_map['AuthHandler'], methods=['POST'])
@sign.signature_required  # This decorator is required for certification.
def auth_request():
    return common_request()

@app.route(route_map['NoAuthHandler'], methods=['POST'])
def no_auth_request():
    return common_request()

def create_response(message, code, success):
    return jsonify(message=message, code=code, success=success)

def common_request():
    start_time = time.time()
    if not request.json or 'image' not in request.json:
        abort(400)

    if interface_manager.total == 0:
        logger.info('There is currently no model deployment and services are not available.')
        return create_response("", -999, False)

    bytes_batch, response = image_utils.get_bytes_batch(request.json['image'])
    if not bytes_batch:
        logger.error('Name[{}] - Response[{}] - {} ms'.format(
            request.json.get('model_site'), response,
            (time.time() - start_time) * 1000)
        )
        return create_response(response, 200, False)

    image_sample = bytes_batch[0]
    image_size = ImageUtils.size_of_image(image_sample)
    size_string = "{}x{}".format(image_size[0], image_size[1])

    if 'model_name' in request.json:
        interface = interface_manager.get_by_name(request.json['model_name'])
    else:
        interface = interface_manager.get_by_size(size_string)

    split_char = request.json.get('output_split', interface.model_conf.output_split)

    if 'need_color' in request.json and request.json['need_color']:
        bytes_batch = [color_extract.separate_color(_, color_map[request.json['need_color']]) for _ in bytes_batch]

    image_batch, response = ImageUtils.get_image_batch(interface.model_conf, bytes_batch)
    if not image_batch:
        logger.error('[{}] - Size[{}] - Name[{}] - Response[{}] - {} ms'.format(
            interface.name, size_string, request.json.get('model_name'), response,
            (time.time() - start_time) * 1000)
        )
        return create_response(response, 200, False)

    result = interface.predict_batch(image_batch, split_char)
    logger.info('[{}] - Size[{}] - Name[{}] - Predict Result[{}] - {} ms'.format(
        interface.name,
        size_string,
        request.json.get('model_name'),
        result,
        (time.time() - start_time) * 1000
    ))
    response['message'] = result
    return create_response(response, 200, True)

def event_loop():
    event = threading.Event()
    observer = Observer()
    event_handler = FileEventHandler(system_config, model_path, interface_manager)
    observer.schedule(event_handler, event_handler.model_conf_path, True)
    observer.start()
    try:
        while True:
            event.wait(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

threading.Thread(target=event_loop, daemon=True).start()

if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option('-p', '--port', type="int", default=19951, dest="port")

    opt, args = parser.parse_args()
    server_port = opt.port
    server_host = "0.0.0.0"

    logger.info('Running on http://{}:{}/ <Press CTRL + C to quit>'.format(server_host, server_port))
    server = WSGIServer((server_host, server_port), app, handler_class=WebSocketHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.stop()