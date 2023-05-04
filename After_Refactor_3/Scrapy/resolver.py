from typing import List

from twisted.internet import defer
from twisted.internet.base import ThreadedResolver
from twisted.internet.interfaces import IHostnameResolver, IHostResolution, IResolutionReceiver, IResolverSimple
from zope.interface.declarations import implementer, provider

from scrapy.utils.datatypes import LocalCache

dnscache = LocalCache(10000)


@implementer(IResolverSimple)
class CachingThreadedResolver(ThreadedResolver):
    def __init__(self, reactor, cache_size: int, timeout: float):
        super().__init__(reactor)
        dnscache.limit = cache_size
        self.timeout = timeout

    @classmethod
    def from_crawler(cls, crawler, reactor):
        cache_size = crawler.settings.getint("DNSCACHE_SIZE") if crawler.settings.getbool("DNSCACHE_ENABLED") else 0
        return cls(reactor, cache_size, crawler.settings.getfloat("DNS_TIMEOUT"))

    def install_on_reactor(self):
        self.reactor.installResolver(self)

    def getHostByName(self, name: str, timeout=None) -> defer.Deferred:
        if name in dnscache:
            return defer.succeed(dnscache[name])

        timeout = (self.timeout,)
        deferred = super().getHostByName(name, timeout)

        if dnscache.limit:
            deferred.addCallback(self._cache_result, name)

        return deferred

    def _cache_result(self, result, name):
        dnscache[name] = result
        return result


@implementer(IHostResolution)
class HostResolution:
    def __init__(self, name: str):
        self.name = name

    def cancel(self):
        raise NotImplementedError()


@provider(IResolutionReceiver)
class _CachingResolutionReceiver:
    def __init__(self, resolutionReceiver, hostName: str):
        self.resolutionReceiver = resolutionReceiver
        self.hostName = hostName
        self.addresses: List[str] = []

    def resolutionBegan(self, resolution):
        self.resolutionReceiver.resolutionBegan(resolution)
        self.resolution = resolution

    def addressResolved(self, address: str):
        self.resolutionReceiver.addressResolved(address)
        self.addresses.append(address)

    def resolutionComplete(self):
        self.resolutionReceiver.resolutionComplete()
        if self.addresses:
            dnscache[self.hostName] = self.addresses


@implementer(IHostnameResolver)
class CachingHostnameResolver:
    def __init__(self, reactor, cache_size: int):
        self.reactor = reactor
        self.original_resolver = reactor.nameResolver
        dnscache.limit = cache_size

    @classmethod
    def from_crawler(cls, crawler, reactor):
        cache_size = crawler.settings.getint("DNSCACHE_SIZE") if crawler.settings.getbool("DNSCACHE_ENABLED") else 0
        return cls(reactor, cache_size)

    def install_on_reactor(self):
        self.reactor.installNameResolver(self)

    def resolveHostName(
        self,
        resolutionReceiver,
        hostName: str,
        portNumber=0,
        addressTypes=None,
        transportSemantics="TCP",
    ):
        try:
            addresses = dnscache[hostName]
        except KeyError:
            return self.original_resolver.resolveHostName(
                _CachingResolutionReceiver(resolutionReceiver, hostName),
                hostName,
                portNumber,
                addressTypes,
                transportSemantics,
            )

        resolutionReceiver.resolutionBegan(HostResolution(hostName))
        for addr in addresses:
            resolutionReceiver.addressResolved(addr)
        resolutionReceiver.resolutionComplete()
        return resolutionReceiver 