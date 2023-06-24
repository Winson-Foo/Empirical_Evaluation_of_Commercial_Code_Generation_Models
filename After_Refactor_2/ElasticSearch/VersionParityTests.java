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
 * Test class for JDBC-ES server version compatibility.
 */
public class VersionCompatibilityTests extends WebServerTestCase {

    private static final String URL_PREFIX = JdbcConfiguration.URL_PREFIX;
    private static final String CONTENT_TYPE_JSON = "application/json";
    private static final String EXCEPTION_MSG_PREFIX = "This version of the JDBC driver is only compatible with Elasticsearch version ";

    /**
     * Ensure that an exception is thrown when connecting to a server with an incompatible version.
     */
    public void testExceptionThrownOnIncompatibleVersions() throws IOException, SQLException {
        String url = URL_PREFIX + webServerAddress();
        Version firstVersion = VersionUtils.getFirstVersion();
        Version serverVersion = Version.V_7_7_0;
        do {
            serverVersion = VersionUtils.getPreviousVersion(serverVersion);
            logger.info("Checking exception is thrown for server version {}", serverVersion);

            prepareVersionResponse(serverVersion);
            String expectedExceptionMsg = getExpectedExceptionMessage(serverVersion);

            SQLException ex = expectThrows(SQLException.class, () -> createJdbcClient(url));
            assertEquals(expectedExceptionMsg, ex.getMessage());
        } while (serverVersion.compareTo(firstVersion) > 0);
    }

    /**
     * Ensure that no exception is thrown when connecting to a server with a compatible version.
     */
    public void testNoExceptionThrownForCompatibleVersions() throws IOException {
        String url = URL_PREFIX + webServerAddress();
        Version serverVersion = Version.CURRENT;
        while (serverVersion.compareTo(Version.V_7_7_0) >= 0) {
            prepareVersionResponse(serverVersion);
            try {
                createJdbcClient(url);
            } catch (SQLException sqle) {
                fail("JDBC driver and Elasticsearch server versions should be compatible. Error: " + sqle);
            }
            serverVersion = VersionUtils.getPreviousVersion(serverVersion);
        }
    }

    /**
     * Return the expected exception message for connecting to a server with the given version.
     */
    private String getExpectedExceptionMessage(Version serverVersion) {
        String serverVersionString = SqlVersion.fromString(serverVersion.toString()).toString();
        return EXCEPTION_MSG_PREFIX + ClientVersion.CURRENT.majorMinorToString() + " or newer; attempting to connect to a server version " + serverVersionString;
    }

    /**
     * Prepare a main response for the given server version.
     */
    private void prepareVersionResponse(Version version) throws IOException {
        MainResponse response = (version == null ? createCurrentVersionMainResponse() : createMainResponse(version));
        webServer().enqueue(new MockResponse()
                .setResponseCode(200)
                .addHeader("Content-Type", CONTENT_TYPE_JSON)
                .setBody(XContentHelper.toXContent(response, XContentType.JSON, false).utf8ToString()));
    }

    /**
     * Create a JDBC client for the given URL.
     */
    private void createJdbcClient(String url) throws SQLException {
        new JdbcHttpClient(new JdbcConnection(JdbcConfiguration.create(url, null, 0), false));
    }
}