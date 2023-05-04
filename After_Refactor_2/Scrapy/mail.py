import logging
import os
from io import BytesIO

from email import encoders as Encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from smtplib import SMTP, SMTP_SSL

from scrapy.utils.misc import arg_to_iter
from scrapy.utils.python import to_bytes


logger = logging.getLogger(__name__)

# Defined in the email.utils module, but undocumented:
# https://github.com/python/cpython/blob/v3.9.0/Lib/email/utils.py#L42
COMMASPACE = ", "


class MailSender:
    def __init__(self, smtphost='localhost', mailfrom='scrapy@localhost', smtpuser=None, smtppass=None, smtpport=25, smtptls=False, smtpssl=False, debug=False):
        self.smtphost = smtphost
        self.smtpport = smtpport
        self.smtpuser = smtpuser
        self.smtppass = smtppass
        self.smtptls = smtptls
        self.smtpssl = smtpssl
        self.mailfrom = mailfrom
        self.debug = debug

    @classmethod
    def from_settings(cls, settings):
        return cls(smtphost=settings['MAIL_HOST'], mailfrom=settings['MAIL_FROM'], smtpuser=settings['MAIL_USER'], smtppass=settings['MAIL_PASS'], smtpport=settings.getint('MAIL_PORT'), smtptls=settings.getbool('MAIL_TLS'), smtpssl=settings.getbool('MAIL_SSL'))

    def create_message(self, to, subject, body, cc=None, attachs=(), mimetype='text/plain', charset=None):
        if attachs:
            msg = MIMEMultipart()
        else:
            msg = MIMEText(body, mimetype, charset)

        to = arg_to_iter(to)
        cc = arg_to_iter(cc)

        msg['From'] = self.mailfrom
        msg['To'] = COMMASPACE.join(to)
        msg['Subject'] = subject
        msg['Date'] = formatdate(localtime=True)

        if cc:
            msg['Cc'] = COMMASPACE.join(cc)

        if attachs:
            text = MIMEText(body, mimetype, charset)
            msg.attach(text)

            for attach_name, mimetype, f in attachs:
                part = MIMEBase(*mimetype.split('/'))
                part.set_payload(f.read())
                Encoders.encode_base64(part)
                part.add_header('Content-Disposition', 'attachment', filename=attach_name)
                msg.attach(part)

        return msg

    def _build_server(self):
        server = SMTP(self.smtphost, self.smtpport)

        if self.smtptls:
            server.starttls()

        if self.smtpuser and self.smtppass:
            server.login(self.smtpuser, self.smtppass)

        return server

    def send(self, to, subject, body, cc=None, attachs=(), mimetype='text/plain', charset=None, send_by=None):
        to = arg_to_iter(to)
        cc = arg_to_iter(cc)

        msg = self.create_message(to, subject, body, cc, attachs, mimetype, charset)

        if self.debug:
            logger.debug('Debug mail sent OK: To=%(to)s Cc=%(cc)s Subject=%(subject)s Attachs=%(n_attachs)d', {'to': to, 'cc': cc, 'subject': subject, 'n_attachs': len(attachs)})
            return

        connection = None

        try:
            if self.smtpssl:
                subject = 'SMTP_SSL'

                if not send_by:
                    server = self._build_server()
                else:
                    server = SMTP_SSL(send_by)

                server.login(self.smtpuser, self.smtppass)
                connection = server.sendmail(self.mailfrom, to + cc, msg.as_string())
                logger.debug('SMTP_SSL mail sent OK: To=%(to)s Cc=%(cc)s Subject="%(subject)s" Attachs=%(n_attachs)d', {'to': to, 'cc': cc, 'subject': subject, 'n_attachs': len(attachs)})
            else:
                subject = 'SMTP'

                if not send_by:
                    server = self._build_server()
                else:
                    server = SMTP(send_by)

                server.login(self.smtpuser, self.smtppass)
                connection = server.sendmail(self.mailfrom, to + cc, msg.as_string())
                logger.debug('SMTP mail sent OK: To=%(to)s Cc=%(cc)s Subject="%(subject)s" Attachs=%(n_attachs)d', {'to': to, 'cc': cc, 'subject': subject, 'n_attachs': len(attachs)})
        except Exception as e:
            logger.error(str(e))
        finally:
            if connection:
                connection.quit()

    def send_self(self, subject, body):
        self.send(to=self.mailfrom, subject=subject, body=body) 