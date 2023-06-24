package com.google.android.exoplayer2.source.smoothstreaming.manifest;

import android.content.Context;
import android.net.Uri;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.android.exoplayer2.testutil.TestUtil;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Unit tests for {@link SsManifestParser}. */
@RunWith(AndroidJUnit4.class)
public final class SsManifestParserTest {

  private static final String SAMPLE_ISMC_1 = "sample_ismc_1";
  private static final String SAMPLE_ISMC_2 = "sample_ismc_2";
  private static final Uri TEST_URI = Uri.parse("https://example.com/test.ismc");

  @Test
  public void testParseSmoothStreamingManifest() throws IOException {
    Context context = ApplicationProvider.getApplicationContext();
    parseSampleManifest(context, SAMPLE_ISMC_1);
    parseSampleManifest(context, SAMPLE_ISMC_2);
  }

  private void parseSampleManifest(Context context, String fileName) throws IOException {
    Uri uri = TEST_URI;
    SsManifestParser parser = new SsManifestParser();
    parser.parse(uri, TestUtil.getInputStream(context, fileName));
  }
} 