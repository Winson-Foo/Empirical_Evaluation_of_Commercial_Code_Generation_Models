package org.elasticsearch.common.blobstore.url;

import java.net.MalformedURLException;

import org.apache.http.ConnectionClosedException;
import org.elasticsearch.common.blobstore.BlobContainer;
import org.elasticsearch.common.blobstore.BlobPath;
import org.elasticsearch.common.blobstore.url.http.URLHttpClient;
import org.elasticsearch.common.blobstore.url.http.URLHttpClientIOException;
import org.elasticsearch.common.blobstore.url.http.URLHttpClientSettings;
import org.elasticsearch.common.network.InetAddresses;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.unit.ByteSizeValue;
import org.elasticsearch.core.TimeValue;
import org.elasticsearch.repositories.blobstore.AbstractBlobContainerRetriesTestCase;
import org.hamcrest.Matcher;
import org.junit.AfterClass;
import org.junit.BeforeClass;

import java.net.InetSocketAddress;
import java.net.SocketTimeoutException;
import java.net.URL;

import static org.hamcrest.Matchers.either;
import static org.hamcrest.Matchers.instanceOf;

public class URLBlobContainerTests extends AbstractBlobContainerRetriesTestCase implements AutoCloseable {
    private static URLHttpClient.Factory factory;
    private static final String HTTP_MAX_RETRIES = "http_max_retries";
    private static final String HTTP_SOCKET_TIMEOUT = "http_socket_timeout";

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
    protected BlobContainer createBlobContainer(
        Integer maxRetries,
        TimeValue readTimeout,
        Boolean disableChunkedEncoding,
        ByteSizeValue bufferSize
    ) {
        Settings.Builder settingsBuilder = Settings.builder();

        if (maxRetries != null) {
            settingsBuilder.put(HTTP_MAX_RETRIES, maxRetries);
        }

        if (readTimeout != null) {
            settingsBuilder.put(HTTP_SOCKET_TIMEOUT, readTimeout);
        }

        try (URLBlobStore urlBlobStore = new URLBlobStore(
            settingsBuilder.build(),
            new URL(new HttpEndpointUtil().getEndpointForServer(httpServer)),
            factory.create(URLHttpClientSettings.fromSettings(settingsBuilder.build())),
            URLHttpClientSettings.fromSettings(settingsBuilder.build())
        )) {
            return urlBlobStore.blobContainer(BlobPath.EMPTY);
        } catch (MalformedURLException e) {
            throw new RuntimeException("Unable to create URLBlobStore", e);
        }
    }

    @Override
    public void close() throws Exception {
        factory.close();
    }

    private static class HttpEndpointUtil {
        private String getEndpointForServer(org.elasticsearch.http.HttpServer httpServer) {
            InetSocketAddress address = httpServer.getAddress();
            return "http://" + InetAddresses.toUriString(address.getAddress()) + ":" + address.getPort() + "/";
        }
    }
}