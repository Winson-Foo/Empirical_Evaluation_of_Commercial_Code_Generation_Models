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

public class VersionParityTests extends WebServerTestCase {
	private static final String URL_PREFIX = JdbcConfiguration.URL_PREFIX;
	private static final String CONTENT_TYPE = "application/json";
	private static final String EXCEPTION_MESSAGE_PREFIX = "This version of the JDBC driver is only compatible with Elasticsearch version ";
	private static final String EXCEPTION_MESSAGE_POSTFIX = " or newer; attempting to connect to a server version ";
	private static final int HTTP_RESPONSE_CODE = 200;
	private static final boolean PRETTY_PRINT = false;
	private static final String ROOT_RESPONSE_BODY = "{}";
	private static final String JSON_UTF8 = XContentType.JSON.mediaTypeWithoutParameters() + "; charset=utf-8";

	public void testExceptionThrownOnIncompatibleVersions() throws IOException, SQLException {
		String url = createJdbcUrl();
		Version firstVersion = VersionUtils.getFirstVersion();
		Version version = Version.V_7_7_0;
		
		do {
			version = VersionUtils.getPreviousVersion(version);
			logger.info("Checking exception is thrown for version {}", version);
			prepareMockServerResponse(version, ROOT_RESPONSE_BODY);
			String versionString = SqlVersion.fromString(version.toString()).toString();

			SQLException ex = expectThrows(SQLException.class, () -> createJdbcClient(url));
			assertExceptionMessage(url, versionString, ex);
		} while (version.compareTo(firstVersion) > 0);
	}

	public void testNoExceptionThrownForCompatibleVersions() throws IOException {
		String url = createJdbcUrl();
		Version version = Version.CURRENT;

		while (isCompatible(version)) {
			prepareMockServerResponse(version, ROOT_RESPONSE_BODY);
			try {
				createJdbcClient(url);
			} catch (SQLException sqle) {
				fail("JDBC driver version and Elasticsearch server version should be compatible. Error: " + sqle);
			}
			version = VersionUtils.getPreviousVersion(version);
		}
	}

	private String createJdbcUrl() {
		return URL_PREFIX + webServerAddress();
	}

	private void prepareMockServerResponse(Version version, String responseBody) throws IOException {
		MainResponse response = createMainResponse(version, responseBody);
		webServer().enqueue(createMockResponse(response));
	}

	private MockResponse createMockResponse(MainResponse response) {
		return new MockResponse()
				.setResponseCode(HTTP_RESPONSE_CODE)
				.addHeader("Content-Type", JSON_UTF8)
				.setBody(XContentHelper.toXContent(response, XContentType.JSON, PRETTY_PRINT).utf8ToString());
	}

	private MainResponse createMainResponse(Version version, String responseBody) {
		MainResponse response = new MainResponse();
		response.setName("test-node");
		response.setVersion(version.toString());
		response.setTagline("You Know, for Search");
		response.setBuildHash("xxxxx");
		response.setBuildDate("2019-01-01T00:00:00.000Z");
		response.setBuildSnapshot(false);
		response.setLuceneVersion("7.5.0");
		response.setFeatures(Collections.emptyMap());
		response.setClusterName("test-cluster");
		response.setClusterUuid(UUID.randomUUID().toString());
		response.setNodeId(UUID.randomUUID().toString());
		response.setNodeName("test-node");

		try {
			byte[] bytes = responseBody.getBytes("UTF-8");
			response.readFrom(XContentType.JSON.xContent().createParser(NamedXContentRegistry.EMPTY, DeprecationHandler.THROW_UNSUPPORTED_OPERATION, bytes, 0, bytes.length));
		} catch (IOException e) {
			throw new RuntimeException("Failed to create main response", e);
		}

		return response;
	}

	private boolean isCompatible(Version version) throws IOException {
		prepareMockServerResponse(version, ROOT_RESPONSE_BODY);
		try {
			createJdbcClient(createJdbcUrl());
			return true;
		} catch (SQLException sqle) {
			return false;
		}
	}

	private void assertExceptionMessage(String url, String versionString, SQLException ex) {
		String expectedExceptionMessage = EXCEPTION_MESSAGE_PREFIX +
				ClientVersion.CURRENT.majorMinorToString() +
				EXCEPTION_MESSAGE_POSTFIX +
				versionString;
		assertEquals(expectedExceptionMessage, ex.getMessage());
	}

	private JdbcHttpClient createJdbcClient(String url) throws SQLException {
		return new JdbcHttpClient(new JdbcConnection(JdbcConfiguration.create(url, null, 0), false));
	}
}