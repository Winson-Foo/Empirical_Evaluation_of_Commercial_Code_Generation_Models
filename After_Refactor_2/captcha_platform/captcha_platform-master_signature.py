#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
from functools import wraps
from constants import ServerType
from utils import *


class InvalidUsage(Exception):
    def __init__(self, message, code=None):
        Exception.__init__(self)
        self.message = message
        self.success = False
        self.code = code

    def to_dict(self):
        rv = {'code': self.code, 'message': self.message, 'success': self.success}
        return rv


class Signature:
    """API signature authentication"""

    def __init__(self, server_type: ServerType, conf: Config):
        self.conf = conf
        self._except = Response(self.conf.response_def_map)
        self._auth = []
        self._timestamp_expiration = 120
        self.request = None
        self.type = server_type

    def set_auth(self, auth):
        self._auth = auth

    def _check_req_timestamp(self, req_timestamp):
        """Check the timestamp"""
        if len(str(req_timestamp)) == 10:
            req_timestamp = int(req_timestamp)
            now_timestamp = SignUtils.timestamp()
            if now_timestamp - self._timestamp_expiration <= req_timestamp <= now_timestamp + self._timestamp_expiration:
                return True
        return False

    def _check_req_access_key(self, req_access_key):
        """Check the access_key in the request parameter"""
        if req_access_key in [i['accessKey'] for i in self._auth if "accessKey" in i]:
            return True
        return False

    def _get_secret_key(self, access_key):
        """Obtain the corresponding secret_key according to access_key"""
        secret_keys = [i['secretKey'] for i in self._auth if i.get('accessKey') == access_key]
        return "" if not secret_keys else secret_keys[0]

    def _sign(self, args):
        """MD5 signature"""
        if "sign" in args:
            args.pop("sign")
        access_key = args["accessKey"]
        query_string = '&'.join(['{}={}'.format(k, v) for (k, v) in sorted(args.items())])
        query_string = '&'.join([query_string, self._get_secret_key(access_key)])
        return SignUtils.md5(query_string).upper()

    def _verification(self, req_params, tornado_handler=None):
        """Verify that the request is valid"""
        try:
            req_signature = req_params.get("sign")
            req_timestamp = req_params.get("timestamp")
            req_access_key = req_params.get("accessKey")

            if None in [req_signature, req_timestamp, req_access_key]:
                raise InvalidUsage(**self._except.INVALID_PUBLIC_PARAMS)
        except Exception:
            raise InvalidUsage(**self._except.UNKNOWN_SERVER_ERROR)

        if self._verify_request(req_signature, req_timestamp, req_access_key, tornado_handler):
            return True

    def _verify_request(self, req_signature, req_timestamp, req_access_key, tornado_handler):
        """Verify the request"""
        if self.type == ServerType.FLASK or self.type == ServerType.SANIC:
            from flask.app import HTTPException, json

            return self._verify_flask_sanic(req_signature, req_timestamp, req_access_key, HTTPException, json)
                
        elif self.type == ServerType.TORNADO:
            from tornado.web import HTTPError

            return self._verify_tornado(req_signature, req_timestamp, req_access_key, HTTPError, tornado_handler)

        raise Exception('Unknown Server Type')

    def _verify_flask_sanic(self, req_signature, req_timestamp, req_access_key, exception, json):
        """Verify the request in Flask or Sanic"""
        if not self._check_req_timestamp(req_timestamp):
            raise exception(response=json.jsonify(self._except.INVALID_TIMESTAMP))

        if not self._check_req_access_key(req_access_key):
            raise exception(response=json.jsonify(self._except.INVALID_ACCESS_KEY))

        if req_signature == self._sign(req_params):
            return True
        else:
            raise exception(response=json.jsonify(self._except.INVALID_QUERY_STRING))

    def _verify_tornado(self, req_signature, req_timestamp, req_access_key, exception, tornado_handler):
        """Verify the request in Tornado"""
        if not self._check_req_timestamp(req_timestamp):
            return tornado_handler.write_error(self._except.INVALID_TIMESTAMP['code'])

        if not self._check_req_access_key(req_access_key):
            return tornado_handler.write_error(self._except.INVALID_ACCESS_KEY['code'])

        if req_signature == self._sign(req_params):
            return True
        else:
            return tornado_handler.write_error(self._except.INVALID_QUERY_STRING['code'])

    def signature_required(self, f):
        """Decorator for signature required"""

        @wraps(f)
        def decorated_function(*args, **kwargs):
            if self.type == ServerType.FLASK:
                from flask import request
                params = request.json
            elif self.type == ServerType.TORNADO:
                from tornado.escape import json_decode
                params = json_decode(args[0].request.body)
            elif self.type == ServerType.SANIC:
                params = args[0].json
            else:
                raise UserWarning('Illegal type, the current version is not supported at this time.')

            result = self._verification(params, args[0] if self.type == ServerType.TORNADO else None)
            if result is True:
                return f(*args, **kwargs)

        return decorated_function