"""
Mail sending helpers

See documentation in docs/topics/email.rst
"""
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.nonmultipart import MIMENonMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
import logging
from typing import Any, List, Optional, Tuple
from io import BytesIO

from scrapy.utils.misc import arg_to_iter
from scrapy.utils.python import to_bytes

from twisted.internet import defer, reactor
from twisted.mail.smtp import ESMTPSenderFactory
from twisted.python.versions import Version

logger = logging.getLogger(__name__)

class MailSender:
    """
    A class for sending email using SMTP.
    """
    def __init__(
            self,
            smtphost: str = "localhost",
            mailfrom: str = "scrapy@localhost",
            smtpuser: Optional[str] = None,
            smtppass: Optional[str] = None,
            smtpport: int = 25,
            smtptls: bool = False,
            smtpssl: bool = False,
            debug: bool = False
    ) -> None:
        self.smtphost = smtphost
        self.smtpport = smtpport
        self.smtpuser = to_bytes(smtpuser) if smtpuser is not None else None
        self.smtppass = to_bytes(smtppass) if smtppass is not None else None
        self.smtptls = smtptls
        self.smtpssl = smtpssl
        self.mailfrom = mailfrom
        self.debug = debug

    @classmethod
    def from_settings(cls, settings: Any) -> 'MailSender':
        """
        Create a MailSender instance from Scrapy settings.
        """
        return cls(
            smtphost=settings["MAIL_HOST"],
            mailfrom=settings["MAIL_FROM"],
            smtpuser=settings["MAIL_USER"],
            smtppass=settings["MAIL_PASS"],
            smtpport=settings.getint("MAIL_PORT"),
            smtptls=settings.getbool("MAIL_TLS"),
            smtpssl=settings.getbool("MAIL_SSL"),
        )

    def send(
            self,
            to: List[str],
            subject: str,
            body: str,
            cc: Optional[List[str]] = None,
            attachs: Optional[List[Tuple[str, str, Any]]] = None,
            mimetype: str = "text/plain",
            charset: Optional[str] = None,
            _callback: Optional[Any] = None
    ) -> Optional[defer.Deferred]:
        """
        Send mail with given parameters.
        """
        if attachs:
            msg = self._attach_files(body, charset, mimetype, attachs)
        else:
            msg = MIMENonMultipart(*mimetype.split("/", 1))
            msg.set_payload(body)

        to = arg_to_iter(to)
        cc = arg_to_iter(cc)

        msg["From"] = self.mailfrom
        msg["To"] = COMMASPACE.join(to)
        msg["Date"] = formatdate(localtime=True)
        msg["Subject"] = subject

        rcpts = list(to)
        if cc:
            rcpts.extend(cc)
            msg["Cc"] = COMMASPACE.join(cc)

        if charset:
            msg.set_charset(charset)

        if _callback:
            _callback(to=to, subject=subject, body=body, cc=cc, attach=attachs, msg=msg)

        if self.debug:
            logger.debug(
                "Debug mail sent OK: To=%(mailto)s Cc=%(mailcc)s "
                'Subject="%(mailsubject)s" Attachs=%(mailattachs)d',
                {
                    "mailto": to,
                    "mailcc": cc,
                    "mailsubject": subject,
                    "mailattachs": len(attachs or []),
                },
            )
            return None

        dfd = self._sendmail(rcpts, msg.as_string().encode(charset or "utf-8"))
        dfd.addCallbacks(
            callback=self._sent_ok,
            errback=self._sent_failed,
            callbackArgs=[to, cc, subject, len(attachs or [])],
            errbackArgs=[to, cc, subject, len(attachs or [])],
        )
        reactor.addSystemEventTrigger("before", "shutdown", lambda: dfd)
        return dfd

    def _attach_files(self, body: str, charset: Optional[str], mimetype: str,
                      attachs: List[Tuple[str, str, Any]]) -> MIMEMultipart:
        msg = MIMEMultipart()
        msg.attach(MIMEText(body, "plain", charset or "us-ascii"))
        for attach_name, mimetype, f in attachs:
            part = MIMEBase(*mimetype.split("/"))
            part.set_payload(f.read())
            Encoders.encode_base64(part)
            part.add_header("Content-Disposition", "attachment", filename=attach_name)
            msg.attach(part)
        msg.set_type(mimetype)
        return msg

    def _sent_ok(self, result: Any, to: List[str], cc: Optional[List[str]], subject: str, nattachs: int) -> None:
        logger.info(
            "Mail sent OK: To=%(mailto)s Cc=%(mailcc)s "
            'Subject="%(mailsubject)s" Attachs=%(mailattachs)d',
            {
                "mailto": to,
                "mailcc": cc,
                "mailsubject": subject,
                "mailattachs": nattachs,
            },
        )

    def _sent_failed(self, failure: Exception, to: List[str], cc: Optional[List[str]], subject: str, nattachs: int) -> Any:
        errstr = str(failure.value)
        logger.error(
            "Unable to send mail: To=%(mailto)s Cc=%(mailcc)s "
            'Subject="%(mailsubject)s" Attachs=%(mailattachs)d'
            "- %(mailerr)s",
            {
                "mailto": to,
                "mailcc": cc,
                "mailsubject": subject,
                "mailattachs": nattachs,
                "mailerr": errstr,
            },
        )
        return failure

    def _sendmail(self, to_addrs: List[str], msg: bytes) -> defer.Deferred:
        msg_io = BytesIO(msg)
        d = defer.Deferred()

        factory = self._create_sender_factory(to_addrs, msg_io, d)

        if self.smtpssl:
            reactor.connectSSL(
                self.smtphost, self.smtpport, factory, ssl.ClientContextFactory()
            )
        else:
            reactor.connectTCP(self.smtphost, self.smtpport, factory)

        return d

    def _create_sender_factory(self, to_addrs: List[str], msg: BytesIO, d: defer.Deferred) -> ESMTPSenderFactory:
        factory_keywords = {
            "heloFallback": True,
            "requireAuthentication": False,
            "requireTransportSecurity": self.smtptls,
        }

        # Newer versions of twisted require the hostname to use STARTTLS
        if twisted_version >= Version("twisted", 21, 2, 0):
            factory_keywords["hostname"] = self.smtphost

        return ESMTPSenderFactory(
            self.smtpuser,
            self.smtppass,
            self.mailfrom,
            to_addrs,
            msg,
            d,
            **factory_keywords
        ) 