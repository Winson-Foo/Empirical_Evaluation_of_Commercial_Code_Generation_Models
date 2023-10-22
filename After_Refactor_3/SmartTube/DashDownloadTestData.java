package com.google.android.exoplayer2.source.dash.offline;

import android.net.Uri;
import com.google.android.exoplayer2.C;
import java.nio.charset.Charset;

/** Data for DASH downloading tests. */
interface DashDownloadTestData {

  String TEST_ID = "test.mpd";
  Uri TEST_MPD_URI = Uri.parse(TEST_ID);

  // XML content stored in separate files
  byte[] TEST_MPD = loadResource("test_mpd.xml");
  byte[] TEST_MPD_NO_INDEX = loadResource("test_mpd_no_index.xml");

  private static byte[] loadResource(String fileName) {
    try (InputStream inputStream = DashDownloadTestData.class.getResourceAsStream(fileName)) {
      byte[] bytes = ByteStreams.toByteArray(inputStream);
      return bytes;
    } catch (IOException e) {
      throw new RuntimeException("Error loading file: " + fileName);
    }
  }

}

