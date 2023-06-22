package org.elasticsearch.common.blobstore.url;

import java.net.InetSocketAddress;
import java.net.MalformedURLException;
import java.net.SocketTimeoutException;
import java.net.URL;

import org.apache.http.ConnectionClosedException;
import org.elasticsearch.common.blobstore.BlobContainer;
import org.elasticsearch.common.blobstore.BlobPath;
import org.elasticsearch.common.blobstore.url.http.URLHttpClient;
import org.elasticsearch.common.blobstore.url.http.URLHttpClientIOException;
import org.elasticsearch.common.blobstore.url.http.URLHttpClientSettings;
import org.elasticsearch.common.network.InetAddresses;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.unit.ByteSizeValue;
import org.elasticsearch.core.SuppressForbidden;
import org.elasticsearch.core.TimeValue;
import org.elasticsearch.repositories.blobstore.AbstractBlobContainerRetriesTestCase;
import org.hamcrest.Matcher;
import org.junit.AfterClass;
import org.junit.BeforeClass;

import static org.hamcrest.Matchers.either;
import static org.hamcrest.Matchers.instanceOf;

@SuppressForbidden(reason = "use a http server")
public class URLBlobStoreIntegrationTests extends AbstractBlobContainerRetriesTestCase {

    private static final String HTTP_MAX_RETRIES_SETTING_KEY = "http_max_retries";
    private static final String HTTP_SOCKET_TIMEOUT_SETTING_KEY = "http_socket_timeout";

    private static URLHttpClient.Factory factory;

    @BeforeClass
    public static void setUpHttpClient() {
        factory = new URLHttpClient.Factory();
    }

    @AfterClass
    public static void tearDownHttpClient() {
        factory.close();
    }

    @Override
    protected String downloadStorageEndpoint(BlobContainer container, String blob) {
        return "/" + container.path().buildAsString() + blob;
    }

    @Override
    protected String bytesContentType() {
        return "application/octet-stream";
    }

    @Override
    protected Class<? extends Exception> unresponsiveExceptionType() {
        return URLHttpClientIOException.class;
    }

    @Override
    protected Matcher<Object> readTimeoutExceptionMatcher() {
        // If the timeout is too tight it's possible that an URLHttpClientIOException is thrown as that
        // exception is thrown before reading data from the response body.
        return either(instanceOf(SocketTimeoutException.class)).or(instanceOf(ConnectionClosedException.class))
                .or(instanceOf(RuntimeException.class))
                .or(instanceOf(URLHttpClientIOException.class));
    }

    @Override
    protected BlobContainer createBlobContainer(Integer maxRetries, TimeValue readTimeout,
            Boolean disableChunkedEncoding, ByteSizeValue bufferSize) {
        final Settings settings = buildSettings(maxRetries, readTimeout);
        final URLHttpClient httpClient = createHttpClient(settings);
        final URLBlobStore urlBlobStore = createUrlBlobStore(settings, httpClient);
        return urlBlobStore.blobContainer(BlobPath.EMPTY);
    }

    private Settings buildSettings(Integer maxRetries, TimeValue readTimeout) {
        final Settings.Builder settingsBuilder = Settings.builder();
        if (maxRetries != null) {
            settingsBuilder.put(HTTP_MAX_RETRIES_SETTING_KEY, maxRetries);
        }
        if (readTimeout != null) {
            settingsBuilder.put(HTTP_SOCKET_TIMEOUT_SETTING_KEY, readTimeout);
        }
        return settingsBuilder.build();
    }

    private URLHttpClient createHttpClient(Settings settings) {
        final URLHttpClientSettings httpClientSettings = URLHttpClientSettings.fromSettings(settings);
        return factory.create(httpClientSettings);
    }

    private URLBlobStore createUrlBlobStore(Settings settings, URLHttpClient httpClient) {
        try {
            final URL endpointUrl = new URL(getEndpointForServer());
            return new URLBlobStore(settings, endpointUrl, httpClient, URLHttpClientSettings.fromSettings(settings));
        } catch (MalformedURLException e) {
            throw new RuntimeException("Unable to create URLBlobStore", e);
        }
    }

    private String getEndpointForServer() {
        final InetSocketAddress address = httpServer.getAddress();
        return "http://" + InetAddresses.toUriString(address.getAddress()) + ":" + address.getPort() + "/";
    }
}