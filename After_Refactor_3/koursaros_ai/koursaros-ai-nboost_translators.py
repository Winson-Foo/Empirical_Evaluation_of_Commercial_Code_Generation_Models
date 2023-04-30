from typing import Any, Dict

import json
import requests
from flask import Response as FlaskResponse
from urllib.parse import ParseResult, urlencode, urlparse, parse_qsl
from werkzeug.local import LocalProxy


DictRequest = Dict[str, Any]
DictResponse = Dict[str, Any]


def remove_unnecessary_headers(headers: Dict[str, str]) -> None:
    headers.pop('content-encoding', None)
    headers.pop('content-length', None)
    headers.pop('transfer-encoding', None)


def flask_request_to_dict_request(flask_request: LocalProxy) -> DictRequest:
    url = urlparse(flask_request.url)

    return {
        'headers': dict(flask_request.headers),
        'method': flask_request.method,
        'url': {
            'scheme': url.scheme,
            'netloc': url.netloc,
            'path': url.path,
            'params': url.params,
            'query': dict(parse_qsl(url.query)),
            'fragment': url.fragment
        },
        'body': dict(flask_request.json) if flask_request.json else {}
    }


def flask_request_to_requests_response(flask_request: LocalProxy) -> requests.Response:
    url = urlparse(flask_request.url)

    return requests.request(
        headers=flask_request.headers,
        method=flask_request.method,
        url=f"{url.scheme}://{url.netloc}{url.path}",
        json=flask_request.json,
    )


def dict_request_to_requests_response(dict_request: DictRequest) -> requests.Response:
    url = ParseResult(
        scheme=dict_request['url']['scheme'],
        netloc=dict_request['url']['netloc'],
        path=dict_request['url']['path'],
        params=dict_request['url']['params'],
        query=urlencode(dict_request['url']['query'], quote_via=lambda x, *a: x),
        fragment=dict_request['url']['fragment']
    ).geturl()

    return requests.request(
        headers=dict_request['headers'],
        method=dict_request['method'],
        url=url,
        json=dict_request['body']
    )


def requests_response_to_dict_response(requests_response: requests.Response) -> DictResponse:
    remove_unnecessary_headers(requests_response.headers)

    return {
        'status': requests_response.status_code,
        'headers': dict(requests_response.headers),
        'body': requests_response.json()
    }


def requests_response_to_flask_response(requests_response: requests.Response) -> FlaskResponse:
    remove_unnecessary_headers(requests_response.headers)

    return FlaskResponse(
        response=requests_response.content,
        status=requests_response.status_code,
        headers=requests_response.headers
    )


def dict_response_to_flask_response(dict_response: DictResponse) -> FlaskResponse:
    return FlaskResponse(
        response=json.dumps(dict_response['body']),
        status=dict_response['status'],
        headers=dict_response['headers'],
    )