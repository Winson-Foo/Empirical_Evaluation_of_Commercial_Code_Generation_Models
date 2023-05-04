from twisted.internet import defer
from twisted.internet.base import ThreadedResolver
from twisted.internet.interfaces import (
    IHostnameResolver,
    IHostResolution,
    IResolutionReceiver,
    IResolverSimple,
)
from zope.interface.declarations import implementer, provider
from scrapy.utils.datatypes import LocalCache

# Initialize dnscache with 10000 entries
dnscache = LocalCache(10000)

@implementer(IResolverSimple)
class CachingThreadedResolver(ThreadedResolver):
    """Default caching resolver. IPv4 only, supports setting a timeout value for DNS requests."""

    def __init__(self, reactor, cache_size, timeout):
        super().__init__(reactor)
        self.timeout = timeout
        dnscache.limit = cache_size

    @classmethod
    def from_crawler(cls, crawler, reactor):
        """Create an instance of CachingThreadedResolver from a Scrapy crawler object and a Twisted reactor."""

        if crawler.settings.getbool("DNSCACHE_ENABLED"):
            cache_size = crawler.settings.getint("DNSCACHE_SIZE")
        else:
            cache_size = 0

        return cls(reactor, cache_size, crawler.settings.getfloat("DNS_TIMEOUT"))

    def install_on_reactor(self):
        """Install the resolver on the reactor."""
        self.reactor.installResolver(self)

    def getHostByName(self, name, timeout=None):
        """Resolve a hostname to an IP address."""

        # Check the cache for the given hostname
        if name in dnscache:
            return defer.succeed(dnscache[name])

        # Use the timeout value from the settings if not specified
        if not timeout:
            timeout = (self.timeout,)

        # Resolve the hostname using the parent class method
        d = super().getHostByName(name, timeout)

        # Add the result to the cache if it's enabled
        if dnscache.limit:
            d.addCallback(self._cache_result, name)

        return d

    def _cache_result(self, result, name):
        """Add a result to the cache."""
        dnscache[name] = result
        return result

@implementer(IHostResolution)
class HostResolution:
    """A placeholder for the result of a hostname resolution."""

    def __init__(self, name):
        self.name = name

    def cancel(self):
        raise NotImplementedError()

@provider(IResolutionReceiver)
class _CachingResolutionReceiver:
    """A resolution receiver that caches the result."""

    def __init__(self, resolutionReceiver, hostName):
        self.resolutionReceiver = resolutionReceiver
        self.hostName = hostName
        self.addresses = []

    def resolutionBegan(self, resolution):
        self.resolutionReceiver.resolutionBegan(resolution)
        self.resolution = resolution

    def addressResolved(self, address):
        self.resolutionReceiver.addressResolved(address)
        self.addresses.append(address)

    def resolutionComplete(self):
        self.resolutionReceiver.resolutionComplete()
        if self.addresses:
            dnscache[self.hostName] = self.addresses

@implementer(IHostnameResolver)
class CachingHostnameResolver:
    """
    Experimental caching resolver. Resolves IPv4 and IPv6 addresses,
    does not support setting a timeout value for DNS requests.
    """

    def __init__(self, reactor, cache_size):
        self.reactor = reactor
        self.original_resolver = reactor.nameResolver
        dnscache.limit = cache_size

    @classmethod
    def from_crawler(cls, crawler, reactor):
        """Create an instance of CachingHostnameResolver from a Scrapy crawler object and a Twisted reactor."""

        if crawler.settings.getbool("DNSCACHE_ENABLED"):
            cache_size = crawler.settings.getint("DNSCACHE_SIZE")
        else:
            cache_size = 0

        return cls(reactor, cache_size)

    def install_on_reactor(self):
        """Install the resolver on the reactor."""
        self.reactor.installNameResolver(self)

    def resolveHostName(self, resolutionReceiver, hostName, portNumber=0, addressTypes=None, transportSemantics="TCP"):
        """Resolve a hostname to an IP address."""

        try:
            addresses = dnscache[hostName]
        except KeyError:
            # Use the original resolver if the hostname is not in the cache
            return self.original_resolver.resolveHostName(_CachingResolutionReceiver(resolutionReceiver, hostName), hostName, portNumber, addressTypes, transportSemantics)

        # Return the cached result
        resolutionReceiver.resolutionBegan(HostResolution(hostName))
        for addr in addresses:
            resolutionReceiver.addressResolved(addr)
        resolutionReceiver.resolutionComplete()
        return resolutionReceiver 