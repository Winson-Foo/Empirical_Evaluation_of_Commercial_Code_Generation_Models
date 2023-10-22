from mimetypes import MimeTypes
from pkgutil import get_data
from scrapy.http import Response
from scrapy.utils.misc import load_object
from scrapy.utils.python import binary_is_text, to_bytes, to_unicode


class MimeTypeLoader:
    def __init__(self):
        self.mime_types = MimeTypes()
        mime_data = get_data("scrapy", "mime.types").decode("utf8")
        self.mime_types.read(fp=StringIO(mime_data))

    def guess_type(self, filename):
        return self.mime_types.guess_type(filename)


class ClassLoader:
    def load_object(self, class_name):
        return load_object(class_name)


class FileTypeGuesser:
    def guess(self, filepath):
        mimetype, encoding = self.loader.guess_type(filepath)
        if mimetype and not encoding:
            return mimetype.split("/", maxsplit=1)[0] + "/*"
        return None

    def __init__(self):
        self.loader = MimeTypeLoader()


class ResponseClassSelector:
    TEXT_RESPONSE_TYPES = {'text/html', 'application/xhtml+xml',
                           'application/vnd.wap.xhtml+xml',
                           'text/*', 'application/json',
                           'application/x-json',
                           'application/json-amazonui-streaming',
                           'application/javascript',
                           'application/x-javascript',
                           'text/xml'}

    def __init__(self, class_loader, file_type_guesser):
        self.response_classes = {
            'text/html': class_loader.load_object('scrapy.http.HtmlResponse'),
            'application/atom+xml': class_loader.load_object('scrapy.http.XmlResponse'),
            'application/rdf+xml': class_loader.load_object('scrapy.http.XmlResponse'),
            'application/rss+xml': class_loader.load_object('scrapy.http.XmlResponse'),
            'application/xml': class_loader.load_object('scrapy.http.XmlResponse'),
            'text/xml': class_loader.load_object('scrapy.http.XmlResponse'),
            'application/json': class_loader.load_object('scrapy.http.TextResponse'),
            'application/x-json': class_loader.load_object('scrapy.http.TextResponse'),
            'application/json-amazonui-streaming': class_loader.load_object('scrapy.http.TextResponse'),
            'application/javascript': class_loader.load_object('scrapy.http.TextResponse'),
            'application/x-javascript': class_loader.load_object('scrapy.http.TextResponse'),
            'text/*': class_loader.load_object('scrapy.http.TextResponse'),
        }
        self.class_loader = class_loader
        self.file_type_guesser = file_type_guesser

    def from_mimetype(self, mimetype):
        """Return the most appropriate Response class for the given mimetype"""
        if mimetype in self.response_classes:
            return self.response_classes[mimetype]
        basetype = mimetype.split("/", maxsplit=1)[0] + "/*"
        if basetype in self.response_classes:
             return self.response_classes[basetype]
        return Response

    def from_filename(self, filename):
        file_type = self.file_type_guesser.guess(filename)
        if file_type:
            return self.from_mimetype(file_type)
        return Response

    def from_content_type(self, content_type, content_encoding=None):
        """Return the most appropriate Response class from an HTTP Content-Type
        header"""
        if content_encoding:
            return Response
        mimetype = to_unicode(content_type).split(";")[0].strip().lower()
        return self.from_mimetype(mimetype)

    def from_content_disposition(self, content_disposition):
        try:
            filename = (
                to_unicode(content_disposition, encoding="latin-1", errors="replace")
                .split(";")[1]
                .split("=")[1]
                .strip("\"'")
            )
            return self.from_filename(filename)
        except IndexError:
            return Response

    def from_headers(self, headers):
        """Return the most appropriate Response class by looking at the HTTP
        headers"""
        cls = Response
        if b"Content-Type" in headers:
            cls = self.from_content_type(
                content_type=headers[b"Content-Type"],
                content_encoding=headers.get(b"Content-Encoding"),
            )
        if cls is Response and b"Content-Disposition" in headers:
            cls = self.from_content_disposition(headers[b"Content-Disposition"])
        return cls

    def from_body(self, body):
        """Try to guess the appropriate response based on the body content.
        This method is a bit magic and could be improved in the future, but
        it's not meant to be used except for special cases where response types
        cannot be guess using more straightforward methods."""
        chunk = body[:5000]
        chunk = to_bytes(chunk)
        if not binary_is_text(chunk):
            return self.from_mimetype("application/octet-stream")
        lowercase_chunk = chunk.lower()
        if b"<html>" in lowercase_chunk:
            return self.from_mimetype("text/html")
        if b"<?xml" in lowercase_chunk:
            return self.from_mimetype("text/xml")
        if b"<!doctype html>" in lowercase_chunk:
            return self.from_mimetype("text/html")
        return self.from_mimetype("text")

    def from_args(self, headers=None, url=None, filename=None, body=None):
        """Guess the most appropriate Response class based on
        the given arguments."""
        cls = Response
        if headers is not None:
            cls = self.from_headers(headers)
        if cls is Response and url is not None:
            cls = self.from_filename(url)
        if cls is Response and filename is not None:
            cls = self.from_filename(filename)
        if cls is Response and body is not None:
            cls = self.from_body(body)
        return cls


class ResponseTypes:
    def __init__(self, class_loader, file_type_guesser, response_class_selector):
        self.class_loader = class_loader
        self.file_type_guesser = file_type_guesser
        self.response_class_selector = response_class_selector

    def from_mimetype(self, mimetype):
        self.response_class_selector.from_mimetype(mimetype)

    def from_content_type(self, content_type, content_encoding=None):
        self.response_class_selector.from_content_type(content_type, content_encoding)

    # other methods

class_loader = ClassLoader()
file_type_guesser = FileTypeGuesser()
response_class_selector = ResponseClassSelector(class_loader, file_type_guesser)
response_types = ResponseTypes(class_loader, file_type_guesser, response_class_selector)