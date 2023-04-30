import traceback
from time import perf_counter
from typing import Dict, List
from json.decoder import JSONDecodeError

from flask import Flask, jsonify, request as flask_request, Response as FlaskResponse, send_from_directory
from werkzeug.routing import Rule

from nboost import defaults, PKG_PATH
from nboost.database import Database
from nboost.delegates import RequestDelegate, ResponseDelegate
from nboost.logger import set_logger
from nboost.plugins import Plugin, resolve_plugin
from nboost.plugins.debug import DebugPlugin
from nboost.plugins.qa.base import QAModelPlugin
from nboost.plugins.rerank.base import RerankModelPlugin
from nboost.translators import (
    dict_request_to_flask_response,
    dict_request_to_requests_response,
    dict_response_to_flask_response,
    requests_response_to_dict_response,
    flask_request_to_dict_request
)


class Proxy:
    def __init__(self,
                 host: str = defaults.host,
                 port: int = defaults.port,
                 verbose: bool = defaults.verbose,
                 no_rerank: bool = defaults.no_rerank,
                 model: str = defaults.model,
                 model_dir: str = defaults.model_dir,
                 qa: bool = defaults.qa,
                 qa_model: str = defaults.qa_model,
                 qa_model_dir: str = defaults.qa_model_dir,
                 frontend_route: str = defaults.frontend_route,
                 status_route: str = defaults.status_route,
                 debug: bool = defaults.debug,
                 **cli_args: str) -> None:
        self.logger = set_logger(self.__class__.__name__, verbose=verbose)
        self.plugins = []  # type: List[Plugin]
        self.db = Database()

        if not no_rerank:  # TODO: FIX SO WORKS WITH DYNAMIC SETTING
            rerank_model_plugin = resolve_plugin(
                model,
                model_dir=model_dir,
                **cli_args
            )  # type: RerankModelPlugin
            self.plugins.append(rerank_model_plugin)

        if qa:
            qa_model_plugin = resolve_plugin(
                qa_model,
                model_dir=qa_model_dir,
                **cli_args
            )  # type: QAModelPlugin
            self.plugins.append(qa_model_plugin)

        if debug:
            debug_plugin = DebugPlugin(**cli_args)
            self.plugins.append(debug_plugin)

        self.static_dir = str(PKG_PATH.joinpath('resources/frontend'))
        self.flask_app = Flask(__name__)
        self.frontend_route = frontend_route
        self.status_route = status_route

        self.url_map_rules = [
            Rule('/<path:path>', endpoint='proxy'),
            Rule('/', defaults={'path': ''})
        ]

        for rule in self.url_map_rules:
            self.flask_app.url_map.add(rule)

    def run(self) -> None:
        self.logger.critical('LISTENING %s:%s' % (host, port))
        self.flask_app.run(host=host, port=port)

    def frontend(self, path: str) -> FlaskResponse:
        if not path:
            path = 'index.html'
        return send_from_directory(self.static_dir, path)

    def status(self) -> FlaskResponse:
        configs = {}
        for plugin in self.plugins:
            configs.update(plugin.configs)

        stats = self.db.get_stats()
        return jsonify({**configs, **stats})

    def proxy_request(self, path: str) -> FlaskResponse:
        dict_request = flask_request_to_dict_request(flask_request)
        db_row = self.db.new_row()
        query_args = {}
        for key in list(dict_request['url']['query']):
            if key in defaults.__dict__:
                query_args[key] = dict_request['url']['query'].pop(key)
        json_args = dict_request['body'].pop('nboost', {})
        args = {**cli_args, **json_args, **query_args}

        request = RequestDelegate(dict_request, **args)
        request.dict['headers'].pop('Host', '')
        request.set_path('url.headers.host', '%s:%s' % (request.uhost, request.uport))
        request.set_path('url.netloc', '%s:%s' % (request.uhost, request.uport))
        request.set_path('url.scheme', 'https' if request.ussl else 'http')

        for plugin in self.plugins:
            plugin.on_request(request, db_row)

        start_time = perf_counter()
        requests_response = dict_request_to_requests_response(dict_request)
        db_row.response_time = perf_counter() - start_time
        try:
            dict_response = requests_response_to_dict_response(requests_response)
        except JSONDecodeError:
            print(requests_response.content)
            return requests_response.content
        response = ResponseDelegate(dict_response, request)
        response.set_path('body.nboost', {})
        db_row.choices = len(response.choices)

        for plugin in self.plugins:
            plugin.on_response(response, db_row)

        self.db.insert(db_row)
        return dict_response_to_flask_response(dict_response)

    def error_handler(self, error: Exception) -> FlaskResponse:
        self.logger.error('', exc_info=True)
        print(traceback.format_exc())
        return jsonify({
            'type': error.__class__.__name__,
            'doc': error.__class__.__doc__,
            'msg': str(error.args)
        }), 500

    def init_routes(self):
        self.flask_app.add_url_rule(self.frontend_route, 'frontend', self.frontend)
        self.flask_app.add_url_rule(self.frontend_route + '/<path:path>', 'frontend_path', self.frontend)
        self.flask_app.add_url_rule(self.frontend_route + self.status_route, 'status_path', self.status)
        self.flask_app.add_url_rule('/', 'proxy_through', self.proxy_request, defaults={'path': ''})
        self.flask_app.register_error_handler(Exception, self.error_handler)
       
if __name__ == '__main__':
    proxy = Proxy()
    proxy.init_routes()
    proxy.run()