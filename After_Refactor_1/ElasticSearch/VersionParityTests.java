package org.elasticsearch.xpack.sql.jdbc;

import java.io.IOException;
import java.sql.SQLException;

import org.elasticsearch.Version;
import org.elasticsearch.common.xcontent.XContentHelper;
import org.elasticsearch.rest.root.MainResponse;
import org.elasticsearch.test.VersionUtils;
import org.elasticsearch.test.http.MockResponse;
import org.elasticsearch.xcontent.XContentType;
import org.elasticsearch.xpack.sql.client.ClientVersion;
import org.elasticsearch.xpack.sql.proto.SqlVersion;

/**
 * Test class for JDBC-ES server version compatibility checks.
 */
public class VersionParityTests extends WebServerTestCase {

    private static final String JDBC_URL_PREFIX = JdbcConfiguration.URL_PREFIX;
    private static final String EXCEPTION_MESSAGE_FORMAT = "This version of the JDBC driver is only compatible with "
            + "Elasticsearch version %s or newer; attempting to connect to a server version %s";

    /**
     * Tests that an SQLException is thrown when the JDBC driver's version is not compatible with the Elasticsearch
     * server's version.
     */
    public void testExceptionThrownOnIncompatibleVersions() throws IOException, SQLException {
        Version current = VersionUtils.getFirstVersion();
        Version serverVersion = Version.V_7_7_0;

        while (serverVersion.compareTo(current) > 0) {
            logger.info("Checking exception is thrown for version {}", serverVersion);
            prepareWebServerResponse(serverVersion);

            String expectedMessage = String.format(EXCEPTION_MESSAGE_FORMAT,
                    ClientVersion.CURRENT.majorMinorToString(), SqlVersion.fromString(serverVersion.toString()));

            SQLException exception = expectThrows(SQLException.class,
                    () -> new JdbcHttpClient(new JdbcConnection(JdbcConfiguration.create(getJdbcUrl(), null, 0), false)));
            assertEquals(expectedMessage, exception.getMessage());

            serverVersion = VersionUtils.getPreviousVersion(serverVersion);
        }
    }

    /**
     * Tests that no SQLException is thrown when the JDBC driver's version is compatible with the Elasticsearch server's
     * version.
     */
    public void testNoExceptionThrownForCompatibleVersions() throws IOException {
        Version serverVersion = Version.CURRENT;

        while (serverVersion.compareTo(Version.V_7_7_0) >= 0) {
            prepareWebServerResponse(serverVersion);
            try {
                new JdbcHttpClient(new JdbcConnection(JdbcConfiguration.create(getJdbcUrl(), null, 0), false));
            } catch (SQLException sqle) {
                fail("JDBC driver version and Elasticsearch server version should be compatible. Error: " + sqle);
            }

            serverVersion = VersionUtils.getPreviousVersion(serverVersion);
        }
    }

    private void prepareWebServerResponse(Version version) throws IOException {
        MainResponse response = version == null ? createCurrentVersionMainResponse() : createMainResponse(version);
        webServer().enqueue(new MockResponse()
                .setResponseCode(200)
                .addHeader("Content-Type", "application/json")
                .setBody(XContentHelper.toXContent(response, XContentType.JSON, false).utf8ToString()));
    }

    private String getJdbcUrl() {
        return JDBC_URL_PREFIX + webServerAddress();
    }
}