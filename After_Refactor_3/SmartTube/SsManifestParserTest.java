/**
 * Copyright (C) 2016 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.android.exoplayer2.source.smoothstreaming.manifest;

import android.net.Uri;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.android.exoplayer2.testutil.TestUtil;
import java.io.IOException;
import java.io.InputStream;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Unit tests for {@link SsManifestParser}. */
@RunWith(AndroidJUnit4.class)
public final class SsManifestParserTest {

  private static final String SAMPLE_ISMC_1 = "sample_ismc_1";
  private static final String SAMPLE_ISMC_2 = "sample_ismc_2";
  private static final Uri TEST_URI = Uri.parse("https://example.com/test.ismc");

  /**
   * Ensures that the sample manifests parse without any exceptions being thrown.
   *
   * @throws IOException if an error occurs parsing the manifest
   */
  @Test
  public void testParseSmoothStreamingManifest() throws IOException {
    SsManifestParser parser = createParser();
    parseSampleManifest(parser, SAMPLE_ISMC_1);
    parseSampleManifest(parser, SAMPLE_ISMC_2);
  }

  /**
   * Creates a new instance of {@link SsManifestParser}.
   *
   * @return the new parser instance
   */
  private SsManifestParser createParser() {
    return new SsManifestParser();
  }

  /**
   * Parses a sample manifest file with the given filename using the provided parser instance.
   *
   * @param parser the parser to use for parsing
   * @param filename the name of the file to parse
   * @throws IOException if an error occurs reading the file or parsing the manifest
   */
  private void parseSampleManifest(SsManifestParser parser, String filename)
      throws IOException {
    try (InputStream inputStream =
        TestUtil.getInputStream(ApplicationProvider.getApplicationContext(), filename)) {
      parser.parse(TEST_URI, inputStream);
    }
  }
} 