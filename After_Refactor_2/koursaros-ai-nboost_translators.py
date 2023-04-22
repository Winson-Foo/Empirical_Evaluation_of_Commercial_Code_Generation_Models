from typing import Dict

from flask import Response as FlaskResponse
from requests import Response as RequestsResponse, request as requests_request
from urllib.parse import urlparse, parse_qsl, urlencode

__all__ = [
    'flask_request_to_dict_request',
    'dict_request_to_requests_response',
    'requests_response_to_dict_response',
    'dict_response_to_flask_response',
    'requests_response_to_flask_response'
]


def flask_request_to_dict_request(request: LocalProxy) -> Dict:
    """Convert Flask request to dict request."""
    parsed_url = urlparse(request.url)
    query_params = dict(parse_qsl(parsed_url.query))

    return {
        'headers': dict(request.headers),
        'method': request.method,
        'url': {
            'scheme': parsed_url.scheme,
            'netloc': parsed_url.netloc,
            'path': parsed_url.path,
            'params': parsed_url.params,
            'query': query_params,
            'fragment': parsed_url.fragment
        },
        'body': dict(request.json) if request.json else {}
    }


def flask_request_to_requests_response(request: LocalProxy) -> RequestsResponse:
    """Convert Flask request to Requests response."""
    parsed_url = urlparse(request.url)

    return requests_request(
        headers=request.headers,
        method=request.method,
        url=ParseResult(
            scheme=parsed_url.scheme,
            netloc=parsed_url.netloc,
            path=parsed_url.path,
            params=parsed_url.params,
            query=parsed_url.query,
            fragment=parsed_url.fragment
        ).geturl()
    )


def dict_request_to_requests_response(request: Dict) -> RequestsResponse:
    """Convert dict request to Requests response."""
    url = request['url']
    headers = request['headers']
    method = request['method']
    body = request['body']
    query_string = urlencode(url.get('query', {}), quote_via=lambda x, *a: x)
    parsed_url = ParseResult(
        scheme=url['scheme'],
        netloc=url['netloc'],
        path=url['path'],
        params=url['params'],
        query=query_string,
        fragment=url['fragment']
    ).geturl()

    return requests_request(
        headers=headers,
        method=method,
        url=parsed_url,
        json=body
    )


def requests_response_to_dict_response(response: RequestsResponse) -> Dict:
    """Convert Requests response to dict response."""
    content = response.content
    status_code = response.status_code
    headers = response.headers
    body = response.json()

    headers.pop('content-encoding', '')
    headers.pop('content-length', '')
    headers.pop('transfer-encoding', '')
    
    return {
        'status': status_code,
        'headers': dict(headers),
        'body': body
    }


def requests_response_to_flask_response(response: RequestsResponse) -> FlaskResponse:
    """Convert Requests response to Flask response."""
    content = response.content
    status_code = response.status_code
    headers = response.headers

    headers.pop('content-encoding', '')
    headers.pop('transfer-encoding', '')
    headers.pop('content-length', '')

    return FlaskResponse(
        response=content,
        status=status_code,
        headers=dict(headers)
    )


def dict_response_to_flask_response(response: Dict) -> FlaskResponse:
    """Convert dict response to Flask response."""
    content = json.dumps(response['body'])
    status_code = response['status']
    headers = response['headers']

    return FlaskResponse(
        response=content,
        status=status_code,
        headers=headers,
    )