from enum import Enum, unique


@unique
class ModelScene(Enum):
    """Enumeration of model scenes"""
    CLASSIFICATION = 'Classification'


@unique
class ModelField(Enum):
    """Enumeration of model fields"""
    IMAGE = 'Image'
    TEXT = 'Text'


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

    def __init__(self, definition_map: dict):
        # SIGN
        self.INVALID_PUBLIC_PARAMS = dict(Message='Invalid Public Params', StatusCode=400001, StatusBool=False)
        self.UNKNOWN_SERVER_ERROR = dict(Message='Unknown Server Error', StatusCode=400002, StatusBool=False)
        self.INVALID_TIMESTAMP = dict(Message='Invalid Timestamp', StatusCode=400004, StatusBool=False)
        self.INVALID_ACCESS_KEY = dict(Message='Invalid Access Key', StatusCode=400005, StatusBool=False)
        self.INVALID_QUERY_STRING = dict(Message='Invalid Query String', StatusCode=400006, StatusBool=False)

        # SERVER
        self.SUCCESS = dict(Message=None, StatusCode=000000, StatusBool=True)
        self.INVALID_IMAGE_FORMAT = dict(Message='Invalid Image Format', StatusCode=500001, StatusBool=False)
        self.INVALID_BASE64_STRING = dict(Message='Invalid Base64 String', StatusCode=500002, StatusBool=False)
        self.IMAGE_DAMAGE = dict(Message='Image Damage', StatusCode=500003, StatusBool=False)
        self.IMAGE_SIZE_NOT_MATCH_GRAPH = dict(Message='Image Size Not Match Graph Value', StatusCode=500004, StatusBool=False)

        self.INVALID_PUBLIC_PARAMS = self.parse(self.INVALID_PUBLIC_PARAMS, definition_map)
        self.UNKNOWN_SERVER_ERROR = self.parse(self.UNKNOWN_SERVER_ERROR, definition_map)
        self.INVALID_TIMESTAMP = self.parse(self.INVALID_TIMESTAMP, definition_map)
        self.INVALID_ACCESS_KEY = self.parse(self.INVALID_ACCESS_KEY, definition_map)
        self.INVALID_QUERY_STRING = self.parse(self.INVALID_QUERY_STRING, definition_map)

        self.SUCCESS = self.parse(self.SUCCESS, definition_map)
        self.INVALID_IMAGE_FORMAT = self.parse(self.INVALID_IMAGE_FORMAT, definition_map)
        self.INVALID_BASE64_STRING = self.parse(self.INVALID_BASE64_STRING, definition_map)
        self.IMAGE_DAMAGE = self.parse(self.IMAGE_DAMAGE, definition_map)
        self.IMAGE_SIZE_NOT_MATCH_GRAPH = self.parse(self.IMAGE_SIZE_NOT_MATCH_GRAPH, definition_map)

    def find_message(self, code):
        error_codes = [value for value in vars(self).values() if isinstance(value, dict)]
        matching_codes = [error['Message'] for error in error_codes if error['StatusCode'] == code]
        return matching_codes[0] if matching_codes else None

    def find(self, code):
        error_codes = [value for value in vars(self).values() if isinstance(value, dict)]
        matching_codes = [error for error in error_codes if error['StatusCode'] == code]
        return matching_codes[0] if matching_codes else None

    def all_codes(self):
        error_codes = [value for value in vars(self).values() if isinstance(value, dict)]
        return [error['Message'] for error in error_codes]

    @staticmethod
    def parse(source: dict, target_map: dict):
        return {target_map[key]: value for key, value in source.items()}