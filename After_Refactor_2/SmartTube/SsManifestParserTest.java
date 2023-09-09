@RunWith(AndroidJUnit4.class)
public final class SsManifestParserTest {

  private static final String SAMPLE_ISMC_1 = "sample_ismc_1";
  private static final String SAMPLE_ISMC_2 = "sample_ismc_2";

  /**
   * Tests that the sample manifests parse without any exceptions being thrown.
   */
  @Test
  public void testParseSmoothStreamingManifest() throws IOException {
    SsManifestParser parser = new SsManifestParser();

    // Parse the first sample manifest
    Uri uri1 = Uri.parse("https://example.com/test1.ismc");
    InputStream inputStream1 = TestUtil.getInputStream(ApplicationProvider.getApplicationContext(), SAMPLE_ISMC_1);
    parseSampleManifest(parser, uri1, inputStream1);

    // Parse the second sample manifest
    Uri uri2 = Uri.parse("https://example.com/test2.ismc");
    InputStream inputStream2 = TestUtil.getInputStream(ApplicationProvider.getApplicationContext(), SAMPLE_ISMC_2);
    parseSampleManifest(parser, uri2, inputStream2);
  }

  /**
   * Parses a single sample manifest using the given parser.
   */
  private void parseSampleManifest(SsManifestParser parser, Uri uri, InputStream inputStream) throws IOException {
    parser.parse(uri, inputStream);
    inputStream.close();
  }
} 