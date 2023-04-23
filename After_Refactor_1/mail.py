import logging
from email import encoders as Encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.nonmultipart import MIMENonMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
from io import BytesIO
from typing import Any, List, Optional, Tuple

from twisted import version as twisted_version
from twisted.internet import defer, reactor, ssl
from twisted.mail.smtp import ESMTPSenderFactory

from scrapy.utils.misc import arg_to_iter
from scrapy.utils.python import to_bytes


logger = logging.getLogger(__name__)
COMMASPACE = ", "


class MailSender:
    smtphost: str
    smtpport: int
    smtpuser: Optional[bytes]
    smtppass: Optional[bytes]
    smtptls: bool
    smtpssl: bool
    mailfrom: str
    debug: bool

    def __init__(
        self,
        smtphost: str = "localhost",
        mailfrom: str = "scrapy@localhost",
        smtpuser: Optional[str] = None,
        smtppass: Optional[str] = None,
        smtpport: int = 25,
        smtptls: bool = False,
        smtpssl: bool = False,
        debug: bool = False,
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
    def from_settings(cls, settings: dict[str, Any]) -> "MailSender":
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
        cc: List[str] = None,
        attachs: List[Tuple[str, str, Any]] = None,
        mimetype: str = "text/plain",
        charset: Optional[str] = None,
        _callback: Optional[Any] = None,
    ) -> Optional[defer.Deferred]:

        to = list(arg_to_iter(to))
        cc = list(arg_to_iter(cc))

        if attachs:
            msg = MIMEMultipart()
        else:
            msg = MIMENonMultipart(*mimetype.split("/", 1))

        msg["From"] = self.mailfrom
        msg["To"] = COMMASPACE.join(to)
        msg["Date"] = formatdate(localtime=True)
        msg["Subject"] = subject

        if cc:
            msg["Cc"] = COMMASPACE.join(cc)

        rcpts = to[:]
        if cc:
            rcpts.extend(cc)

        if charset:
            msg.set_charset(charset)

        if attachs:
            msg.attach(MIMEText(body, "plain", charset or "us-ascii"))
            for attach_name, mimetype, f in attachs:
                part = MIMEBase(*mimetype.split("/"))
                part.set_payload(f.read())
                Encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition", "attachment", filename=attach_name
                )
                msg.attach(part)
        else:
            msg.set_payload(body)

        if _callback:
            _callback(to=to, subject=subject, body=body, cc=cc, attach=attachs, msg=msg)

        if self.debug:
            logger.debug(
                f"Debug mail sent OK: To={to} Cc={cc} Subject=\"{subject}\" "
                f"Attachs={len(attachs)}"
            )
            return None

        dfd = self._sendmail(rcpts, msg.as_string().encode(charset or "utf-8"))
        dfd.addCallbacks(
            callback=self._sent_ok,
            errback=self._sent_failed,
            callbackArgs=[to, cc, subject, len(attachs)],
            errbackArgs=[to, cc, subject, len(attachs)],
        )

        reactor.addSystemEventTrigger("before", "shutdown", lambda: dfd)
        return dfd

    def _sent_ok(
        self, result: Any, to: List[str], cc: List[str], subject: str, nattachs: int
    ) -> None:
        logger.info(
            f"Mail sent OK: To={to} Cc={cc} Subject={subject} " f"Attachs={nattachs}"
        )

    def _sent_failed(
        self,
        failure: Any,
        to: List[str],
        cc: List[str],
        subject: str,
        nattachs: int,
    ) -> Any:
        errstr = str(failure.value)
        logger.error(
            f"Unable to send mail: To={to} Cc={cc} Subject={subject} "
            f"Attachs={nattachs}- {errstr}"
        )
        return failure

    def _sendmail(
        self, to_addrs: List[str], msg: bytes
    ) -> defer.Deferred:
        msg_stream = BytesIO(msg)
        d = defer.Deferred()
        factory = self._create_sender_factory(to_addrs, msg_stream, d)

        if self.smtpssl:
            reactor.connectSSL(
                self.smtphost,
                self.smtpport,
                factory,
                ssl.ClientContextFactory(),
            )
        else:
            reactor.connectTCP(self.smtphost, self.smtpport, factory)

        return d

    def _create_sender_factory(
        self, to_addrs: List[str], msg: BytesIO, d: defer.Deferred
    ) -> ESMTPSenderFactory:

        factory_kwargs = {
            "heloFallback": True,
            "requireAuthentication": False,
            "requireTransportSecurity": self.smtptls,
        }

        if twisted_version >= Version("twisted", 21, 2, 0):
            factory_kwargs["hostname"] = self.smtphost

        factory = ESMTPSenderFactory(
            self.smtpuser,
            self.smtppass,
            self.mailfrom,
            to_addrs,
            msg,
            d,
            **factory_kwargs
        )

        factory.noisy = False
        return factory