from functools import wraps
from flask import Flask, request, jsonify
from tornado.escape import json_decode
from tornado.web import HTTPError
from constants import ServerType, Response
from utils import SignUtils

app = Flask(__name__)
response_def_map = {}


class InvalidUsage(Exception):

    def __init__(self, message, code=None):
        super().__init__(self, message)
        self.code = code
        self.success = False

    def to_dict(self):
        return {'code': self.code, 'message': self.message, 'success': self.success}


def create_app(server_type: ServerType):
    return app


def init_auth(auth):
    return auth


def timestamp_expiration():
    return 120


def get_secret_key(auth, access_key):
    secret_keys = [i['secretKey'] for i in auth if i.get('accessKey') == access_key]
    return "" if not secret_keys else secret_keys[0]


class Signature:

    def __init__(self, server_type: ServerType, auth):
        self.auth = init_auth(auth)
        self.except = Response(response_def_map)
        self.timestamp_expiration = timestamp_expiration()
        self.server_type = server_type

    def _check_req_timestamp(self, req_timestamp):
        now_timestamp = SignUtils.timestamp()
        return now_timestamp - self.timestamp_expiration <= req_timestamp <= now_timestamp + self.timestamp_expiration

    def _check_req_access_key(self, req_access_key):
        return req_access_key in [i['accessKey'] for i in self.auth if "accessKey" in i]

    def _sign(self, args):
        access_key = args["accessKey"]
        query_string = '&'.join(['{}={}'.format(k, v) for (k, v) in sorted(args.items()) if k != "sign"])
        query_string = '&'.join([query_string, get_secret_key(self.auth, access_key)])
        return SignUtils.md5(query_string).upper()

    def _verification(self, req_params, tornado_handler=None):
        try:
            req_signature = req_params["sign"]
            req_timestamp = req_params["timestamp"]
            req_access_key = req_params["accessKey"]
        except KeyError:
            raise InvalidUsage(**self.except.INVALID_PUBLIC_PARAMS)
        except Exception:
            raise InvalidUsage(**self.except.UNKNOWN_SERVER_ERROR)

        if self.server_type == ServerType.FLASK:
            if not self._check_req_timestamp(req_timestamp):
                raise HTTPError(response=jsonify(self.except.INVALID_TIMESTAMP))
            if not self._check_req_access_key(req_access_key):
                raise HTTPError(response=jsonify(self.except.INVALID_ACCESS_KEY))
            if req_signature != self._sign(req_params):
                raise HTTPError(response=jsonify(self.except.INVALID_QUERY_STRING))
        elif self.server_type == ServerType.TORNADO:
            if not self._check_req_timestamp(req_timestamp):
                return tornado_handler.write_error(self.except.INVALID_TIMESTAMP['code'])
            if not self._check_req_access_key(req_access_key):
                return tornado_handler.write_error(self.except.INVALID_ACCESS_KEY['code'])
            if req_signature != self._sign(req_params):
                return tornado_handler.write_error(self.except.INVALID_QUERY_STRING['code'])
        else:
            raise Exception('Unknown Server Type')

        return True

    def signature_required(self, f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            params = {}
            if self.server_type == ServerType.FLASK:
                params = request.json
            elif self.server_type == ServerType.TORNADO:
                params = json_decode(args[0].request.body)
            elif self.server_type == ServerType.SANIC:
                params = args[0].json
            else:
                raise UserWarning('Illegal type, the current version is not supported at this time.')

            result = self._verification(params, args[0] if self.server_type == ServerType.TORNADO else None)
            if result:
                return f(*args, **kwargs)

        return decorated_function


@app.route("/api/endpoint", methods=["POST"])
@signature.signature_required
def endpoint():
    # Your endpoint logic here
    return jsonify({"success": True})