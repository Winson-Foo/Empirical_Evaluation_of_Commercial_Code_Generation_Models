from typing import List
from twisted.internet import defer
from twisted.internet.base import ThreadedResolver, ResolutionError
from twisted.internet.interfaces import (
    IHostnameResolver,
    IHostResolution,
    IResolutionReceiver,
    IResolverSimple,
)
from zope.interface.declarations import implementer, provider
from scrapy.utils.datatypes import LocalCache


class DNSCache:
    """
    DNS caching implementation using a local cache object.
    """

    def __init__(self, limit: int) -> None:
        self.cache = LocalCache(limit)

    def get(self, key: str) -> List[str]:
        """
        Returns the cached value for the given key, or raises a KeyError if not found.
        """
        return self.cache[key]

    def put(self, key: str, value: List[str]) -> None:
        """
        Puts the value into the cache for the given key.
        """
        self.cache[key] = value


@implementer(IResolverSimple)
class CachingThreadedResolver(ThreadedResolver):
    """
    Default caching resolver. IPv4 only, supports setting a timeout value for DNS requests.
    """

    def __init__(self, reactor, cache_size: int, timeout: float) -> None:
        super().__init__(reactor)
        self.dns_cache = DNSCache(cache_size)
        self.timeout = timeout

    @classmethod
    def from_crawler(cls, crawler, reactor):
        """
        Creates a new resolver from the crawler settings.
        """
        cache_enabled = crawler.settings.getbool("DNSCACHE_ENABLED")
        cache_size = crawler.settings.getint("DNSCACHE_SIZE") if cache_enabled else 0
        timeout = crawler.settings.getfloat("DNS_TIMEOUT")
        return cls(reactor, cache_size, timeout)

    def install_on_reactor(self):
        """
        Installs this resolver on the reactor.
        """
        self.reactor.installResolver(self)

    def getHostByName(self, name: str, timeout: float = None) -> defer.Deferred:
        """
        Resolves the hostname using the cache if available, or fallbacks to the original resolver.
        """
        try:
            return defer.succeed(self.dns_cache.get(name))
        except KeyError:
            # In Twisted<=16.6, getHostByName() is always called with a default timeout of 60s,
            # so we override the input argument here to enforce Scrapy's DNS_TIMEOUT setting value.
            timeout = (self.timeout,)
            d = super().getHostByName(name, timeout)
            d.addCallbacks(self._cache_result, self._handle_error, callbackArgs=(name,))
            return d

    def _cache_result(self, result: List[str], name: str) -> List[str]:
        """
        Caches the DNS resolution result for the given hostname.
        """
        self.dns_cache.put(name, result)
        return result

    def _handle_error(self, failure):
        """
        Handles any resolution errors by raising a more descriptive exception.
        """
        failure.trap(ResolutionError)
        raise ValueError(f"DNS resolution error: {failure.getErrorMessage()}")


@implementer(IHostResolution)
class HostResolution:
    """
    Dummy implementation of host resolution cancellation.
    """

    def __init__(self, name):
        self.name = name

    def cancel(self) -> None:
        pass


@provider(IResolutionReceiver)
class _CachingResolutionReceiver:
    """
    Resolution receiver implementation that stores the resolved addresses into the DNS cache.
    """

    def __init__(self, resolution_receiver, host_name: str):
        self.resolution_receiver = resolution_receiver
        self.host_name = host_name
        self.addresses = []

    def resolutionBegan(self, resolution) -> None:
        self.resolution_receiver.resolutionBegan(resolution)

    def addressResolved(self, address: str) -> None:
        self.resolution_receiver.addressResolved(address)
        self.addresses.append(address)

    def resolutionComplete(self) -> None:
        self.resolution_receiver.resolutionComplete()
        if self.addresses:
            dns_cache.put(self.host_name, self.addresses)


@implementer(IHostnameResolver)
class CachingHostnameResolver:
    """
    Experimental caching resolver. Resolves IPv4 and IPv6 addresses,
    does not support setting a timeout value for DNS requests.
    """

    def __init__(self, reactor, cache_size: int) -> None:
        self.reactor = reactor
        self.original_resolver = reactor.nameResolver
        self.dns_cache = DNSCache(cache_size)

    @classmethod
    def from_crawler(cls, crawler, reactor):
        """
        Creates a new resolver from the crawler settings.
        """
        cache_enabled = crawler.settings.getbool("DNSCACHE_ENABLED")
        cache_size = crawler.settings.getint("DNSCACHE_SIZE") if cache_enabled else 0
        return cls(reactor, cache_size)

    def install_on_reactor(self) -> None:
        """
        Installs this resolver on the reactor.
        """
        self.reactor.installNameResolver(self)

    def resolveHostName(
        self,
        resolution_receiver: IResolutionReceiver,
        host_name: str,
        port_number: int = 0,
        address_types: List[int] = None,
        transport_semantics: str = "TCP",
    ) -> IResolutionReceiver:
        """
        Resolves the hostname using the cache if available, or fallbacks to the original resolver.
        """
        try:
            addresses = self.dns_cache.get(host_name)
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