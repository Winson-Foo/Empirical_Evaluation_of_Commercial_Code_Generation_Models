#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
from enum import Enum, unique


@unique
class ModelScene(Enum):
    """Model scenes enumeration"""
    Classification = 'Classification'


@unique
class ModelField(Enum):
    """Model fields enumeration"""
    Image = 'Image'
    Text = 'Text'


class SystemConfig:
    split_flag = b'\x99\x99\x99\x00\xff\xff\xff\x00\x99\x99\x99'
    default_route = [
        {
            "Class": "AuthHandler",
            "Route": "/captcha/auth/v2"
        },
        {
            "Class": "NoAuthHandler",
            "Route": "/captcha/v1"
        },
        {
            "Class": "SimpleHandler",
            "Route": "/captcha/v3"
        },
        {
            "Class": "HeartBeatHandler",
            "Route": "/check_backend_active.html"
        },
        {
            "Class": "HeartBeatHandler",
            "Route": "/verification"
        },
        {
            "Class": "HeartBeatHandler",
            "Route": "/"
        },
        {
            "Class": "ServiceHandler",
            "Route": "/service/info"
        },
        {
            "Class": "FileHandler",
            "Route": "/service/logs/(.*)",
            "Param": {"path": "logs"}
        },
        {
            "Class": "BaseHandler",
            "Route": ".*"
        }
    ]
    default_config = {
        "System": {
            "DefaultModel": "default",
            "SplitFlag": b'\x99\x99\x99\x00\xff\xff999999.........99999\xff\x00\x99\x99\x99',
            "SavePath": "",
            "RequestCountInterval": 86400,
            "GlobalRequestCountInterval": 86400,
            "RequestLimit": -1,
            "GlobalRequestLimit": -1,
            "WithoutLogger": False,
            "RequestSizeLimit": {},
            "DefaultPort": 19952,
            "IllegalTimeMessage": "The maximum number of requests has been exceeded.",
            "ExceededMessage": "Illegal access time, please request in open hours.",
            "BlacklistTriggerTimes": -1,
            "Whitelist": False,
            "ErrorMessage": {
                400: "Bad Request",
                401: "Unicode Decode Error",
                403: "Forbidden",
                404: "404 Not Found",
                405: "Method Not Allowed",
                500: "Internal Server Error"
            }
        },
        "RouteMap": default_route,
        "Security": {
            "AccessKey": "",
            "SecretKey": ""
        },
        "RequestDef": {
            "InputData": "image",
            "ModelName": "model_name",
        },
        "ResponseDef": {
            "Message": "message",
            "StatusCode": "code",
            "StatusBool": "success",
            "Uid": "uid",
        }
    }


class ServerType(str):
    FLASK = 19951
    TORNADO = 19952
    SANIC = 19953


class Response:

    def __init__(self, def_map: dict):
        def_dict = {
            'INVALID_PUBLIC_PARAMS': ('Invalid Public Params', 400001, False)
            'UNKNOWN_SERVER_ERROR': ('Unknown Server Error', 400002, False),
            'INVALID_TIMESTAMP': ('Invalid Timestamp', 400004, False),
            'INVALID_ACCESS_KEY': ('Invalid Access Key', 400005, False),
            'INVALID_QUERY_STRING': ('Invalid Query String', 400006, False),
            'SUCCESS': (None, 000000, True),
            'INVALID_IMAGE_FORMAT': ('Invalid Image Format', 500001, False),
            'INVALID_BASE64_STRING': ('Invalid Base64 String', 500002, False),
            'IMAGE_DAMAGE': ('Image Damage', 500003, False),
            'IMAGE_SIZE_NOT_MATCH_GRAPH': ('Image Size Not Match Graph Value', 500004, False),
        }
        for name, (message, code, success) in def_dict.items():
            setattr(self, name, self.parse({'Message': message, 'StatusCode': code, 'StatusBool': success}, def_map))

    def find_message(self, code):
        for attr in vars(self).values():
            if attr['StatusCode'] == code:
                return attr['Message']
        return None

    def find(self, code):
        for attr in vars(self).values():
            if attr['StatusCode'] == code:
                return attr
        return None

    def all_code(self):
        return [attr['Message'] for attr in vars(self).values()]

    @staticmethod
    def parse(src: dict, target_map: dict):
        return {target_map[k]: v for k, v in src.items()}