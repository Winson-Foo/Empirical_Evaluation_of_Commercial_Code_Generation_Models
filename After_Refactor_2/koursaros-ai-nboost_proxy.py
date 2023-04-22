from time import perf_counter
from typing import List
import traceback

from flask import Flask, jsonify, request as flask_request, send_from_directory
from json.decoder import JSONDecodeError
from werkzeug.routing import Rule

from nboost import defaults, PKG_PATH
from nboost.database import Database
from nboost.logger import set_logger
from nboost.plugins import Plugin, resolve_plugin
from nboost.plugins.debug import DebugPlugin
from nboost.plugins.qa.base import QAModelPlugin
from nboost.plugins.rerank.base import RerankModelPlugin
from nboost.translators import (
    dict_request_to_flask_request,
    dict_request_to_requests_response,
    flask_request_to_dict_request,
    requests_response_to_dict_response,
)
from nboost.delegates import (
    RequestDelegate,
    ResponseDelegate,
)


class Proxy:
    def __init__(
        self,
        host: type(defaults.host) = defaults.host,
        port: type(defaults.port) = defaults.port,
        verbose: type(defaults.verbose) = defaults.verbose,
        no_rerank: type(defaults.no_rerank) = defaults.no_rerank,
        model: type(defaults.model) = defaults.model,
        model_dir: type(defaults.model_dir) = defaults.model_dir,
        qa: type(defaults.qa) = defaults.qa,
        qa_model: type(defaults.qa_model) = defaults.qa_model,
        qa_model_dir: type(defaults.qa_model_dir) = defaults.qa_model_dir,
        frontend_route: type(defaults.frontend_route) = defaults.frontend_route,
        status_route: type(defaults.status_route) = defaults.status_route,
        debug: type(defaults.debug) = defaults.debug,
        **cli_args,
    ):
        self.logger = set_logger(self.__class__.__name__, verbose=verbose)
        self.db = Database()

        self.plugins = []  # type: List[Plugin]

        if not no_rerank:
            self.add_rerank_model_plugin(model, model_dir, **cli_args)

        if qa:
            self.add_qa_model_plugin(qa_model, qa_model_dir, **cli_args)

        if debug:
            self.add_debug_plugin(**cli_args)

        self.flask_app = Flask(__name__)
        self.static_dir = str(PKG_PATH.joinpath("resources/frontend"))

        self.add_routes(frontend_route, status_route)
        

    def add_routes(self, frontend_route, status_route):
        self.flask_app.route(frontend_route, methods=['GET'])(self.frontend_root)
        self.flask_app.route(frontend_route + '/<path:path>', methods=['GET'])(self.frontend_path)
        self.flask_app.route(frontend_route + status_route)(self.status_path)
        self.flask_app.url_map.add(Rule('/<path:path>', endpoint='proxy'))
        self.flask_app.route('/', defaults={'path': ''})(self.proxy_through)

    def add_qa_model_plugin(self, qa_model, qa_model_dir, **cli_args):
        qa_model_plugin = resolve_plugin(qa_model, model_dir=qa_model_dir, **cli_args)
        self.plugins.append(qa_model_plugin)

    def add_rerank_model_plugin(self, model, model_dir, **cli_args):
        rerank_model_plugin = resolve_plugin(model, model_dir=model_dir, **cli_args)
        self.plugins.append(rerank_model_plugin)

    def add_debug_plugin(self, **cli_args):
        debug_plugin = DebugPlugin(**cli_args)
        self.plugins.append(debug_plugin)

    def frontend_root(self):
        return send_from_directory(self.static_dir, "index.html")

    def frontend_path(self, path):
        return send_from_directory(self.static_dir, path)

    def status_path(self):
        configs = {}

        for plugin in self.plugins:
            configs.update(plugin.configs)

        stats = self.db.get_stats()
        return jsonify({**configs, **stats})

    def proxy_through(self, path):
        dict_request = flask_request_to_dict_request(flask_request)

        db_row = self.db.new_row()

        query_args = {}
        for key in list(dict_request["url"]["query"]):
            if key in defaults.__dict__:
                query_args[key] = dict_request["url"]["query"].pop(key)

        json_args = dict_request["body"].pop("nboost", {})
        args = {**cli_args, **json_args, **query_args}

        flask_request = dict_request_to_flask_request(dict_request)
        request = RequestDelegate(flask_request, **args)

        request.dict["headers"].pop("Host", "")
        request.set_path("url.headers.host", f"{request.uhost}:{request.uport}")
        request.set_path("url.netloc", f"{request.uhost}:{request.uport}")
        request.set_path("url.scheme", "https" if request.ussl else "http")

        for plugin in self.plugins:
            plugin.on_request(request, db_row)

        start_time = perf_counter()
        response = self.get_response(dict_request, args, db_row)
        db_row.response_time = perf_counter() - start_time

        db_row.choices = len(response.choices)

        for plugin in self.plugins:
            plugin.on_response(response, db_row)

        self.db.insert(db_row)
        return dict_response_to_flask_response(response.dict_response)

    @staticmethod
    def get_response(dict_request, args, db_row):
        requests_response = dict_request_to_requests_response(dict_request)
        
        try:
             dict_response = requests_response_to_dict_response(requests_response)
        except JSONDecodeError as ex:
            print(requests_response.content)
            raise ex
        
        response = ResponseDelegate(dict_response, dict_request_to_flask_request(dict_request))
        response.set_path("body.nboost", {})
        return response


    def run(self):
        self.logger.critical(f"LISTENING {host}:{port}")
        return self.flask_app.run(host=host, port=port)
