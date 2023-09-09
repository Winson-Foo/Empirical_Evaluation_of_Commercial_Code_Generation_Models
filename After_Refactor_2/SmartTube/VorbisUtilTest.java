package com.google.android.exoplayer2.extractor.ogg;

import androidx.test.ext.junit.runners.AndroidJUnit4;

import com.google.android.exoplayer2.ParserException;
import com.google.android.exoplayer2.testutil.OggTestData;
import com.google.android.exoplayer2.util.ParsableByteArray;

import org.junit.Test;
import org.junit.runner.RunWith;

import static com.google.android.exoplayer2.extractor.ogg.VorbisUtil.iLog;
import static com.google.android.exoplayer2.extractor.ogg.VorbisUtil.readVorbisCommentHeader;
import static com.google.android.exoplayer2.extractor.ogg.VorbisUtil.readVorbisIdentificationHeader;
import static com.google.android.exoplayer2.extractor.ogg.VorbisUtil.readVorbisModes;
import static com.google.android.exoplayer2.extractor.ogg.VorbisUtil.verifyVorbisHeaderCapturePattern;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

/** Unit tests for {@link VorbisUtil}. */
@RunWith(AndroidJUnit4.class)
public final class VorbisUtilTest {

  private static final int ZERO = 0;
  private static final int ONE = 1;
  private static final int TWO = 2;
  private static final int THREE = 3;
  private static final int FOUR = 4;
  private static final int FIVE = 5;
  private static final int NEGATIVE_ONE = -1;
  private static final int APPROXIMATE_BITRATE = 66666;
  private static final int HEADER_TYPE = 0x01;
  private static final String MAGIC_PATTERN = "vorbis";
  private static final String COMMENT_VENDOR = "Xiph.Org libVorbis I 20120203 (Omnipresent)";
  private static final String COMMENT_ALBUM = "ALBUM=äö";
  private static final String COMMENT_TITLE = "TITLE=A sample song";
  private static final String COMMENT_ARTIST = "ARTIST=Google";

  @Test
  public void testILog() {
    assertThat(iLog(ZERO)).isEqualTo(ZERO);
    assertThat(iLog(ONE)).isEqualTo(ONE);
    assertThat(iLog(TWO)).isEqualTo(TWO);
    assertThat(iLog(THREE)).isEqualTo(TWO);
    assertThat(iLog(FOUR)).isEqualTo(THREE);
    assertThat(iLog(FIVE)).isEqualTo(THREE);
    assertThat(iLog(ONE << FOUR)).isEqualTo(FOUR);
    assertThat(iLog(NEGATIVE_ONE)).isEqualTo(ZERO);
    assertThat(iLog(-122)).isEqualTo(ZERO);
  }

  @Test
  public void testReadIdHeader() {
    byte[] data = OggTestData.getIdentificationHeaderData();
    ParsableByteArray headerData = new ParsableByteArray(data, data.length);
    VorbisUtil.VorbisIdHeader vorbisIdHeader = readVorbisIdentificationHeader(headerData);

    assertThat(vorbisIdHeader.sampleRate).isEqualTo(22050);
    assertThat(vorbisIdHeader.version).isEqualTo(ZERO);
    assertThat(vorbisIdHeader.framingFlag).isTrue();
    assertThat(vorbisIdHeader.channels).isEqualTo(TWO);
    assertThat(vorbisIdHeader.blockSize0).isEqualTo(512);
    assertThat(vorbisIdHeader.blockSize1).isEqualTo(1024);
    assertThat(vorbisIdHeader.bitrateMax).isEqualTo(NEGATIVE_ONE);
    assertThat(vorbisIdHeader.bitrateMin).isEqualTo(NEGATIVE_ONE);
    assertThat(vorbisIdHeader.bitrateNominal).isEqualTo(APPROXIMATE_BITRATE);
    assertThat(vorbisIdHeader.getApproximateBitrate()).isEqualTo(APPROXIMATE_BITRATE);
  }

  @Test
  public void testReadCommentHeader() throws ParserException {
    byte[] data = OggTestData.getCommentHeaderDataUTF8();
    ParsableByteArray headerData = new ParsableByteArray(data, data.length);
    VorbisUtil.CommentHeader commentHeader = readVorbisCommentHeader(headerData);

    assertThat(commentHeader.vendor).isEqualTo(COMMENT_VENDOR);
    assertThat(commentHeader.comments).hasLength(THREE);
    assertThat(commentHeader.comments[ZERO]).isEqualTo(COMMENT_ALBUM);
    assertThat(commentHeader.comments[ONE]).isEqualTo(COMMENT_TITLE);
    assertThat(commentHeader.comments[TWO]).isEqualTo(COMMENT_ARTIST);
  }

  @Test
  public void testReadVorbisModes() throws ParserException {
    byte[] data = OggTestData.getSetupHeaderData();
    ParsableByteArray headerData = new ParsableByteArray(data, data.length);
    VorbisUtil.Mode[] modes = readVorbisModes(headerData, TWO);

    assertThat(modes).hasLength(TWO);
    assertThat(modes[ZERO].blockFlag).isFalse();
    assertThat(modes[ZERO].mapping).isEqualTo(ZERO);
    assertThat(modes[ZERO].transformType).isEqualTo(ZERO);
    assertThat(modes[ZERO].windowType).isEqualTo(ZERO);
    assertThat(modes[ONE].blockFlag).isTrue();
    assertThat(modes[ONE].mapping).isEqualTo(ONE);
    assertThat(modes[ONE].transformType).isEqualTo(ZERO);
    assertThat(modes[ONE].windowType).isEqualTo(ZERO);
  }

  @Test
  public void testVerifyVorbisHeaderCapturePattern() throws ParserException {
    ParsableByteArray header = new ParsableByteArray(
        new byte[] {HEADER_TYPE, 'v', 'o', 'r', 'b', 'i', 's'});
    assertThat(verifyVorbisHeaderCapturePattern(HEADER_TYPE, header, false)).isTrue();
  }

  @Test
  public void testVerifyVorbisHeaderCapturePatternInvalidHeader() {
    ParsableByteArray header = new ParsableByteArray(
        new byte[] {HEADER_TYPE, 'v', 'o', 'r', 'b', 'i', 's'});
    try {
      verifyVorbisHeaderCapturePattern(0x99, header, false);
      fail();
    } catch (ParserException e) {
      assertThat(e.getMessage()).isEqualTo("expected header type 99");
    }
  }

  @Test
  public void testVerifyVorbisHeaderCapturePatternInvalidHeaderQuiet() throws ParserException {
    ParsableByteArray header = new ParsableByteArray(
        new byte[] {HEADER_TYPE, 'v', 'o', 'r', 'b', 'i', 's'});
    assertThat(verifyVorbisHeaderCapturePattern(0x99, header, true)).isFalse();
  }

  @Test
  public void testVerifyVorbisHeaderCapturePatternInvalidPattern() {
    ParsableByteArray header = new ParsableByteArray(
        new byte[] {HEADER_TYPE, 'x', 'v', 'o', 'r', 'b', 'i', 's'});
    try {
      verifyVorbisHeaderCapturePattern(HEADER_TYPE, header, false);
      fail();
    } catch (ParserException e) {
      assertThat(e.getMessage()).isEqualTo("expected characters 'vorbis'");
    }
  }

  @Test
  public void testVerifyVorbisHeaderCapturePatternQuietInvalidPatternQuiet()
      throws ParserException {
    ParsableByteArray header = new ParsableByteArray(
        new byte[] {HEADER_TYPE, 'x', 'v', 'o', 'r', 'b', 'i', 's'});
    assertThat(verifyVorbisHeaderCapturePattern(HEADER_TYPE, header, true)).isFalse();
  }

}

