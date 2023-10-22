package com.google.android.exoplayer2.source.dash.offline;

import android.net.Uri;
import com.google.android.exoplayer2.C;
import java.nio.charset.Charset;

/** Data for DASH downloading tests. */
/* package */ final class DashDownloadTestData {

  private static final String TEST_ID = "test.mpd";
  private static final Uri TEST_MPD_URI = Uri.parse(TEST_ID);
  private static final String TEST_MPD_XML = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
          + "<MPD xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" type=\"static\" "
          + "    mediaPresentationDuration=\"PT31S\">\n"
          + "    <Period duration=\"PT16S\" >\n"
          + "        <AdaptationSet>\n"
          + "            <SegmentList>\n"
          + "                <SegmentTimeline>\n"
          + "                    <S d=\"5\" />\n"
          + "                    <S d=\"5\" />\n"
          + "                    <S d=\"5\" />\n"
          + "                </SegmentTimeline>\n"
          + "            </SegmentList>\n"
          + "            <Representation>\n"
          + "                <SegmentList>\n"
          // Bounded range data
          + "                    <Initialization\n"
          + "                        range=\"0-9\" sourceURL=\"audio_init_data\" />\n"
          // Unbounded range data
          + "                    <SegmentURL media=\"audio_segment_1\" />\n"
          + "                    <SegmentURL media=\"audio_segment_2\" />\n"
          + "                    <SegmentURL media=\"audio_segment_3\" />\n"
          + "                </SegmentList>\n"
          + "            </Representation>\n"
          + "        </AdaptationSet>\n"
          + "        <AdaptationSet>\n"
          // This segment list has a 1 second offset to make sure the progressive download order
          + "            <SegmentList>\n"
          + "                <SegmentTimeline>\n"
          + "                    <S t=\"1\" d=\"5\" />\n" // 1s offset
          + "                    <S d=\"5\" />\n"
          + "                    <S d=\"5\" />\n"
          + "                </SegmentTimeline>\n"
          + "            </SegmentList>\n"
          + "            <Representation>\n"
          + "                <SegmentList>\n"
          + "                    <SegmentURL media=\"text_segment_1\" />\n"
          + "                    <SegmentURL media=\"text_segment_2\" />\n"
          + "                    <SegmentURL media=\"text_segment_3\" />\n"
          + "                </SegmentList>\n"
          + "            </Representation>\n"
          + "        </AdaptationSet>\n"
          + "    </Period>\n"
          + "    <Period>\n"
          + "        <SegmentList>\n"
          + "            <SegmentTimeline>\n"
          + "                <S d=\"5\" />\n"
          + "                <S d=\"5\" />\n"
          + "                <S d=\"5\" />\n"
          + "            </SegmentTimeline>\n"
          + "        </SegmentList>\n"
          + "        <AdaptationSet>\n"
          + "            <Representation>\n"
          + "                <SegmentList>\n"
          + "                    <SegmentURL media=\"period_2_segment_1\" />\n"
          + "                    <SegmentURL media=\"period_2_segment_2\" />\n"
          + "                    <SegmentURL media=\"period_2_segment_3\" />\n"
          + "                </SegmentList>\n"
          + "            </Representation>\n"
          + "        </AdaptationSet>\n"
          + "    </Period>\n"
          + "</MPD>";
  private static final byte[] TEST_MPD = TEST_MPD_XML.getBytes(Charset.forName(C.UTF8_NAME));

  private static final byte[] TEST_MPD_NO_INDEX =
      ("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
              + "<MPD xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" type=\"dynamic\">\n"
              + "    <Period start=\"PT6462826.784S\" >\n"
              + "        <AdaptationSet>\n"
              + "            <Representation>\n"
              + "                <SegmentBase indexRange='0-10'/>\n"
              + "            </Representation>\n"
              + "        </AdaptationSet>\n"
              + "    </Period>\n"
              + "</MPD>")
          .getBytes(Charset.forName(C.UTF8_NAME));

  private DashDownloadTestData() {}

  public static String getTestId() {
    return TEST_ID;
  }

  public static Uri getTestMpdUri() {
    return TEST_MPD_URI;
  }

  public static byte[] getTestMpd() {
    return TEST_MPD;
  }

  public static byte[] getTestMpdNoIndex() {
    return TEST_MPD_NO_INDEX;
  }
} 