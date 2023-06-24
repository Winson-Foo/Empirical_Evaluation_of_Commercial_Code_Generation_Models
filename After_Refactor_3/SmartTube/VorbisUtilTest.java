package com.google.android.exoplayer2.extractor.ogg;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.android.exoplayer2.ParserException;
import com.google.android.exoplayer2.testutil.OggTestData;
import com.google.android.exoplayer2.util.ParsableByteArray;
import org.junit.Test;
import org.junit.runner.RunWith;

import static com.google.common.truth.Truth.assertThat;
import static com.google.android.exoplayer2.extractor.ogg.VorbisUtil.verifyVorbisHeaderCapturePattern;

/** Unit test for {@link VorbisUtil}. */
@RunWith(AndroidJUnit4.class)
public final class VorbisUtilTest {

  @Test
  public void iLog_returnsCorrectValue() throws Exception {
    assertThat(VorbisUtil.iLog(0)).isEqualTo(0);
    assertThat(VorbisUtil.iLog(1)).isEqualTo(1);
    assertThat(VorbisUtil.iLog(2)).isEqualTo(2);
    assertThat(VorbisUtil.iLog(3)).isEqualTo(2);
    assertThat(VorbisUtil.iLog(4)).isEqualTo(3);
    assertThat(VorbisUtil.iLog(5)).isEqualTo(3);
    assertThat(VorbisUtil.iLog(8)).isEqualTo(4);
    assertThat(VorbisUtil.iLog(-1)).isEqualTo(0);
    assertThat(VorbisUtil.iLog(-122)).isEqualTo(0);
  }

  @Test
  public void readVorbisCommentHeader_parsesUtf8Comments() throws Exception {
    byte[] data = OggTestData.getCommentHeaderDataUTF8();
    ParsableByteArray headerData = new ParsableByteArray(data, data.length);

    VorbisUtil.CommentHeader commentHeader =
        VorbisUtil.readVorbisCommentHeader(headerData);

    assertThat(commentHeader.vendor).isEqualTo("Xiph.Org libVorbis I 20120203 (Omnipresent)");

    assertThat(commentHeader.comments)
        .asList()
        .containsExactly("ALBUM=äö", "TITLE=A sample song", "ARTIST=Google");
  }

  @Test
  public void readVorbisIdentificationHeader_parsesDataCorrectly() throws Exception {
    byte[] data = OggTestData.getIdentificationHeaderData();
    ParsableByteArray headerData = new ParsableByteArray(data, data.length);

    VorbisUtil.VorbisIdHeader vorbisIdHeader =
        VorbisUtil.readVorbisIdentificationHeader(headerData);

    assertThat(vorbisIdHeader.sampleRate).isEqualTo(22050);
    assertThat(vorbisIdHeader.version).isEqualTo(0);
    assertThat(vorbisIdHeader.framingFlag).isTrue();
    assertThat(vorbisIdHeader.channels).isEqualTo(2);
    assertThat(vorbisIdHeader.blockSize0).isEqualTo(512);
    assertThat(vorbisIdHeader.blockSize1).isEqualTo(1024);
    assertThat(vorbisIdHeader.bitrateMax).isEqualTo(-1);
    assertThat(vorbisIdHeader.bitrateMin).isEqualTo(-1);
    assertThat(vorbisIdHeader.bitrateNominal).isEqualTo(66666);
    assertThat(vorbisIdHeader.getApproximateBitrate()).isEqualTo(66666);
  }

  @Test
  public void readVorbisModes_parsesModesCorrectly() throws Exception {
    byte[] data = OggTestData.getSetupHeaderData();
    ParsableByteArray headerData = new ParsableByteArray(data, data.length);

    VorbisUtil.Mode[] modes = VorbisUtil.readVorbisModes(headerData, 2);

    assertThat(modes)
        .asList()
        .containsExactly(
            new VorbisUtil.Mode(false, 0, 0, 0),
            new VorbisUtil.Mode(true, 1, 0, 0));
  }

  @Test
  public void verifyVorbisHeaderCapturePattern_parsesHeadersCorrectly() throws Exception {
    ParsableByteArray header =
        new ParsableByteArray(new byte[] {0x01, 'v', 'o', 'r', 'b', 'i', 's'});

    assertThat(verifyVorbisHeaderCapturePattern(0x01, header, false)).isTrue();
  }

  @Test
  public void verifyVorbisHeaderCapturePattern_throwsErrorForInvalidHeader() {
    ParsableByteArray header =
        new ParsableByteArray(new byte[] {0x01, 'v', 'o', 'r', 'b', 'i', 's'});

    try {
      VorbisUtil.verifyVorbisHeaderCapturePattern(0x99, header, false);
    } catch (ParserException e) {
      assertThat(e.getMessage()).isEqualTo("expected header type 99");
    }
  }

  @Test
  public void verifyVorbisHeaderCapturePattern_throwsErrorForInvalidPattern() {
    ParsableByteArray header =
        new ParsableByteArray(new byte[] {0x01, 'x', 'v', 'o', 'r', 'b', 'i', 's'});

    try {
      VorbisUtil.verifyVorbisHeaderCapturePattern(0x01, header, false);
    } catch (ParserException e) {
      assertThat(e.getMessage()).isEqualTo("expected characters 'vorbis'");
    }
  }

  @Test
  public void verifyVorbisHeaderCapturePattern_returnsFalseForInvalidPatternQuietly()
      throws Exception {
    ParsableByteArray header =
        new ParsableByteArray(new byte[] {0x01, 'x', 'v', 'o', 'r', 'b', 'i', 's'});

    assertThat(verifyVorbisHeaderCapturePattern(0x01, header, true)).isFalse();
  }

  @Test
  public void verifyVorbisHeaderCapturePattern_returnsFalseForInvalidHeaderQuite()
      throws Exception {
    ParsableByteArray header =
        new ParsableByteArray(new byte[] {0x01, 'v', 'o', 'r', 'b', 'i', 's'});

    assertThat(verifyVorbisHeaderCapturePattern(0x99, header, true)).isFalse();
  }
} 