# interfaces.py
from typing import Dict

class Request:
    headers: Dict[str,str]
    method: str
    url: str
    body: Dict[str,any]

class Response:
    status: int
    headers: Dict[str,str]
    body: Dict[str,any]


# flask_request_converter.py
from urllib.parse import ParseResult, urlparse, parse_qsl, urlencode
from werkzeug.local import LocalProxy
from interfaces import Request

def flask_request_to_dict_request(flask_request: LocalProxy) -> Request:
    """Convert flask request to dict request."""
    urllib_url = urlparse(flask_request.url)

    return Request(
        headers=dict(flask_request.headers),
        method=flask_request.method,
        url={
            'scheme': urllib_url.scheme,
            'netloc': urllib_url.netloc,
            'path': urllib_url.path,
            'params': urllib_url.params,
            'query': dict(parse_qsl(urllib_url.query)),
            'fragment': urllib_url.fragment
        },
        body=dict(flask_request.json) if flask_request.json else {}
    )


# requests_response_converter.py
import json
from typing import Union
from requests import Response as RequestsResponse
from urllib.parse import urlparse, urlencode, ParseResult
from interfaces import Response, Request

def dict_request_to_requests_response(dict_request: Request) -> RequestsResponse:
    return requests_request(
        headers=dict_request.headers,
        method=dict_request.method,
        url=ParseResult(
            scheme=dict_request.url['scheme'],
            netloc=dict_request.url['netloc'],
            path=dict_request.url['path'],
            params=dict_request.url['params'],
            query=urlencode(dict_request.url['query'], quote_via=lambda x, *a: x),
            fragment=dict_request.url['fragment']
        ).geturl(),
        json=dict_request.body
    )

def requests_response_to_dict_response(requests_response: RequestsResponse) -> Response:
    requests_response.headers.pop('content-encoding', '')
    requests_response.headers.pop('content-length', '')
    requests_response.headers.pop('transfer-encoding', '')
    return Response(
        status=requests_response.status_code,
        headers=dict(requests_response.headers),
        body=requests_response.json()
    )


# flask_response_converter.py
from flask import Response as FlaskResponse
from interfaces import Response

def dict_response_to_flask_response(dict_response: Response) -> FlaskResponse:
    return FlaskResponse(
        response=json.dumps(dict_response.body),
        status=dict_response.status,
        headers=dict_response.headers,
    )

def requests_response_to_flask_response(requests_response: RequestsResponse) -> FlaskResponse:
    requests_response.headers.pop('content-encoding', '')
    requests_response.headers.pop('transfer-encoding', '')
    requests_response.headers.pop('content-length', '')
    return dict_response_to_flask_response(requests_response_to_dict_response(requests_response))

We have created three separate modules for converting requests and responses. Each module contains functions for converting one type of request/response to another. We have also defined Request and Response classes in interfaces.py, which are used by all the modules.

To use these modules, we can do the following:

from flask_request_converter import flask_request_to_dict_request
from requests_response_converter import dict_request_to_requests_response, requests_response_to_dict_response
from flask_response_converter import dict_response_to_flask_response

# convert a Flask request to a dict request
dict_request = flask_request_to_dict_request(flask_request)

# send the dict request to an API
requests_response = dict_request_to_requests_response(dict_request)

# convert the requests response to a dict response
dict_response = requests_response_to_dict_response(requests_response)

# convert the dict response to a Flask response
flask_response = dict_response_to_flask_response(dict_response)