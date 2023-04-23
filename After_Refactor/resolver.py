from typing import List

from twisted.internet import defer
from twisted.internet.base import ThreadedResolver
from twisted.internet.interfaces import IHostnameResolver, IHostResolution, IResolutionReceiver
from zope.interface.declarations import implementer, provider

from scrapy.utils.datatypes import LocalCache


class CachingThreadedResolver(ThreadedResolver):
    """Default caching resolver. IPv4 only, supports setting a timeout value for DNS requests."""

    def __init__(self, reactor, cache_size: int, timeout: float):
        super().__init__(reactor)
        self.dnscache = LocalCache(cache_size)
        self.timeout = timeout

    @classmethod
    def from_crawler(cls, crawler, reactor):
        cache_enabled = crawler.settings.getbool("DNSCACHE_ENABLED")
        cache_size = crawler.settings.getint("DNSCACHE_SIZE") if cache_enabled else 0
        timeout = crawler.settings.getfloat("DNS_TIMEOUT")
        return cls(reactor, cache_size, timeout)

    def install_on_reactor(self):
        with self.reactor:
            self.reactor.installResolver(self)

    def getHostByName(self, name: str, timeout: float = None) -> defer.Deferred:
        if name in self.dnscache:
            return defer.succeed(self.dnscache[name])

        timeout = (self.timeout,) if timeout is None else (timeout,)
        d = super().getHostByName(name, timeout)
        if self.dnscache.limit:
            d.addCallback(self._cache_result, name)
        return d

    def _cache_result(self, result: List[str], name: str) -> List[str]:
        self.dnscache[name] = result
        return result


@implementer(IHostResolution)
class HostResolution:
    """Simple implementation of IHostResolution interface."""

    def __init__(self, name: str):
        self.name = name

    def cancel(self):
        raise NotImplementedError()


@provider(IResolutionReceiver)
class _CachingResolutionReceiver:
    """Internal implementation of IResolutionReceiver for resolving host names."""

    def __init__(self, resolution_receiver: IResolutionReceiver, host_name: str):
        self.resolution_receiver = resolution_receiver
        self.host_name = host_name
        self.addresses = []

    def resolutionBegan(self, resolution: IHostResolution):
        self.resolution_receiver.resolutionBegan(resolution)
        self.resolution = resolution

    def addressResolved(self, address: str):
        self.resolution_receiver.addressResolved(address)
        self.addresses.append(address)

    def resolutionComplete(self):
        self.resolution_receiver.resolutionComplete()
        if self.addresses:
            self.dnscache[self.host_name] = self.addresses


@implementer(IHostnameResolver)
class CachingHostnameResolver:
    """Experimental caching resolver. Resolves IPv4 and IPv6 addresses, does not support setting a timeout value for DNS requests."""

    def __init__(self, reactor, cache_size: int):
        self.reactor = reactor
        self.original_resolver = reactor.nameResolver
        self.dnscache = LocalCache(cache_size)

    @classmethod
    def from_crawler(cls, crawler, reactor):
        cache_enabled = crawler.settings.getbool("DNSCACHE_ENABLED")
        cache_size = crawler.settings.getint("DNSCACHE_SIZE") if cache_enabled else 0
        return cls(reactor, cache_size)

    def install_on_reactor(self):
        with self.reactor:
            self.reactor.installNameResolver(self)

    def resolveHostName(
        self,
        resolution_receiver: IResolutionReceiver,
        host_name: str,
        port_number: int = 0,
        address_types: List[str] = None,
        transport_semantics: str = "TCP",
    ) -> IResolutionReceiver:
        try:
            addresses = self.dnscache[host_name]
        except KeyError:
            return self.original_resolver.resolveHostName(
                _CachingResolutionReceiver(resolution_receiver, host_name),
                host_name,
                port_number,
                address_types,
                transport_semantics,
            )
        else:
            resolution_receiver.resolutionBegan(HostResolution(host_name))
            for addr in addresses:
                resolution_receiver.addressResolved(addr)
            resolution_receiver.resolutionComplete()
            return resolution_receiver