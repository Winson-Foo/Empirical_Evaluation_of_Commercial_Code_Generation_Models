from time import perf_counter
from typing import List
from flask import (
    request as flask_request,
    Response as FlaskResponse,
    send_from_directory,
    jsonify,
    Flask)
from nboost.plugins.rerank.base import RerankModelPlugin
from nboost.delegates import RequestDelegate, ResponseDelegate
from nboost.plugins.qa.base import QAModelPlugin
from nboost.plugins import resolve_plugin
from nboost.plugins.debug import DebugPlugin
from nboost import defaults, PKG_PATH
from nboost.database import Database
from nboost.logger import set_logger
from nboost.plugins import Plugin
from nboost.translators import *
from json.decoder import JSONDecodeError
from werkzeug.routing import Rule
import traceback

class Proxy:
    def __init__(self, host: type(defaults.host) = defaults.host,
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
                 **cli_args):
        self.logger = set_logger(self.__class__.__name__, verbose=verbose)
        db = Database()
        plugins = []  # type: List[Plugin]

        if not no_rerank:  # TODO: FIX SO WORKS WITH DYNAMIC SETTING
            rerank_model_plugin = resolve_plugin(
                model,
                model_dir=model_dir,
                **cli_args)  # type: RerankModelPlugin

            plugins.append(rerank_model_plugin)

        if qa:
            qa_model_plugin = resolve_plugin(
                qa_model,
                model_dir=qa_model_dir,
                **cli_args)  # type: QAModelPlugin

            plugins.append(qa_model_plugin)

        if debug:
            debug_plugin = DebugPlugin(**cli_args)
            plugins.append(debug_plugin)

        static_dir = str(PKG_PATH.joinpath('resources/frontend'))
        flask_app = Flask(__name__)

        @flask_app.route(frontend_route, methods=['GET'])
        def frontend_root():
            return send_from_directory(static_dir, 'index.html')

        @flask_app.route(frontend_route + '/<path:path>', methods=['GET'])
        def frontend_path(path):
            return send_from_directory(static_dir, path)

        @flask_app.route(frontend_route + status_route)
        def status_path():
            configs = {}

            for plugin in plugins:
                configs.update(plugin.configs)

            stats = db.get_stats()
            return jsonify({**configs, **stats})

        flask_app.url_map.add(Rule('/<path:path>', endpoint='proxy'))

        @flask_app.route('/', defaults={'path': ''})
        @flask_app.endpoint('proxy')
        def proxy_through(path):
            # parse the client request
            dict_request = flask_request_to_dict_request(flask_request)  # takes the json

            """Search request."""
            db_row = db.new_row()

            # combine command line args and runtime args sent by request
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

            for plugin in plugins:  # type: Plugin
                plugin.on_request(request, db_row)

            # get response from upstream server
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

            for plugin in plugins:  # type: Plugin
                plugin.on_response(response, db_row)

            # save stats to sql lite
            db.insert(db_row)

            return dict_response_to_flask_response(dict_response)

        @flask_app.errorhandler(Exception)
        def handle_json_response(error):
            self.logger.error('', exc_info=True)
            print(traceback.format_exc())
            return jsonify({
                'type': error.__class__.__name__,
                'doc': error.__class__.__doc__,
                'msg': str(error.args)
            }), 500

        self.run = lambda: (
            self.logger.critical('LISTENING %s:%s' % (host, port)) or
            flask_app.run(host=host, port=port)
        )
