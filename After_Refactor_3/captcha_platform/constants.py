class ServerType:
    FLASK = "FLASK"
    TORNADO = "TORNADO"
    SANIC = "SANIC"


class Response:
    def __init__(self, response_def_map):
        self.INVALID_PUBLIC_PARAMS = response_def_map.get('INVALID_PUBLIC_PARAMS')
        self.UNKNOWN_SERVER_ERROR = response_def_map.get('UNKNOWN_SERVER_ERROR')
        self.INVALID_TIMESTAMP = response_def_map.get('INVALID_TIMESTAMP')
        self.INVALID_ACCESS_KEY = response_def_map.get('INVALID_ACCESS_KEY')
        self.INVALID_QUERY_STRING = response_def_map.get('INVALID_QUERY_STRING')