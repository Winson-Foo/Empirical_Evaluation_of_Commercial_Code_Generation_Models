from time import perf_counter
from typing import List
import traceback
import json

from flask import Flask, Response, request, send_from_directory, jsonify
from werkzeug.routing import Rule

from nboost import defaults, PKG_PATH
from nboost.database import Database
from nboost.delegates import RequestDelegate, ResponseDelegate
from nboost.logger import set_logger
from nboost.plugins import resolve_plugin
from nboost.plugins.debug import DebugPlugin
from nboost.plugins.qa.base import QAModelPlugin
from nboost.plugins.rerank.base import RerankModelPlugin
from nboost.plugins.base import Plugin
from nboost.translators import dict_response_to_flask_response, dict_request_to_requests_response, requests_response_to_dict_response


class Proxy:
    def __init__(self, host: str = defaults.host,
                 port: int = defaults.port, **kwargs):
        self.logger = set_logger(self.__class__.__name__, verbose=kwargs.get('verbose', False))
        self.db = Database()
        self.plugins = []  # type: List[Plugin]

        self._add_plugins(kwargs)

        self.static_dir = str(PKG_PATH.joinpath('resources/frontend'))
        self.app = Flask(__name__)

        self._add_routes()

    def run(self):
        host, port = self.app.config['HOST'], self.app.config['PORT']
        self.logger.critical(f'LISTENING {host}:{port}')
        self.app.run(host=host, port=port)

    def _add_plugins(self, kwargs):
        if not kwargs.get('no_rerank', False):
            rerank_model_plugin = resolve_plugin(kwargs.get('model', defaults.model),
                                                 model_dir=kwargs.get('model_dir', defaults.model_dir),
                                                 **kwargs)
            self.plugins.append(rerank_model_plugin)

        if kwargs.get('qa', False):
            qa_model_plugin = resolve_plugin(kwargs.get('qa_model', defaults.qa_model),
                                             model_dir=kwargs.get('qa_model_dir', defaults.qa_model_dir),
                                             **kwargs)
            self.plugins.append(qa_model_plugin)

        if kwargs.get('debug', False):
            debug_plugin = DebugPlugin(**kwargs)
            self.plugins.append(debug_plugin)

    def _add_routes(self):
        frontend_route = defaults.frontend_route
        status_route = defaults.status_route

        @self.app.route(f'{frontend_route}', methods=['GET'])
        def frontend_root():
            return send_from_directory(self.static_dir, 'index.html')

        @self.app.route(f'{frontend_route}/<path:path>', methods=['GET'])
        def frontend_path(path):
            return send_from_directory(self.static_dir, path)

        @self.app.route(f'{frontend_route}{status_route}')
        def status_path():
            configs = {}
            for plugin in self.plugins:
                configs.update(plugin.configs)

            stats = self.db.get_stats()
            return jsonify({**configs, **stats})

        self.app.url_map.add(Rule('/<path:path>', endpoint='proxy'))

        @self.app.route('/', defaults={'path': ''})
        @self.app.endpoint('proxy')
        def proxy_through(path):
            try:
                dict_request = request.json
            except json.JSONDecodeError:
                self.logger.error('Unable to parse JSON in request body.')
                return Response('Unable to parse JSON.', 400)

            db_row = self.db.new_row()

            query_args = {}
            for key in request.args:
                if key in defaults.__dict__:
                    query_args[key] = request.args.get(key)

            json_args = dict_request.pop('nboost', {})
            args = {**kwargs, **json_args, **query_args}

            request_delegate = RequestDelegate(dict_request, **args)
            request_delegate.dict['headers'].pop('Host', '')
            request_delegate.set_path('url.headers.host', f"{request_delegate.uhost}:{request_delegate.uport}")
            request_delegate.set_path('url.netloc', f"{request_delegate.uhost}:{request_delegate.uport}")
            request_delegate.set_path('url.scheme', 'https' if request_delegate.ussl else 'http')

            for plugin in self.plugins:
                plugin.on_request(request_delegate, db_row)

            start_time = perf_counter()
            resp = dict_request_to_requests_response(dict_request)
            db_row.response_time = perf_counter() - start_time
            try:
                dict_response = requests_response_to_dict_response(resp)
            except json.JSONDecodeError:
                self.logger.error(f"Unable to parse JSON in response body from {resp.url}.")
                return Response(resp.content, status=resp.status_code)

            response_delegate = ResponseDelegate(dict_response, request_delegate)
            response_delegate.set_path('body.nboost', {})
            db_row.choices = len(response_delegate.choices)

            for plugin in self.plugins:
                plugin.on_response(response_delegate, db_row)

            self.db.insert(db_row)
            return dict_response_to_flask_response(dict_response)

        @self.app.errorhandler(Exception)
        def handle_error(e):
            self.logger.error('', exc_info=True)
            traceback.print_exc()
            return jsonify({
                'type': e.__class__.__name__,
                'doc': e.__class__.__doc__,
                'msg': str(e.args)
            }), 500