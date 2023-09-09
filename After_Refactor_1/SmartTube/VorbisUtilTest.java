package com.google.android.exoplayer2.extractor.ogg;

import static com.google.android.exoplayer2.extractor.ogg.VorbisUtil.verifyVorbisHeaderCapturePattern;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.jupiter.api.Assertions.fail;

import com.google.android.exoplayer2.ParserException;
import com.google.android.exoplayer2.testutil.OggTestData;
import com.google.android.exoplayer2.util.ParsableByteArray;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

@DisplayName("VorbisUtil")
public final class VorbisUtilTest {

  @Nested
  @DisplayName("When calculating the iLog")
  public class ILogTests {

    @Test
    @DisplayName("Given a value of 0")
    public void iLog_whenZero_returnsZero() {
      assertThat(VorbisUtil.iLog(0)).isEqualTo(0);
    }

    @Test
    @DisplayName("Given a value of 1")
    public void iLog_whenOne_returnsOne() {
      assertThat(VorbisUtil.iLog(1)).isEqualTo(1);
    }

    @Test
    @DisplayName("Given a value of 2")
    public void iLog_whenTwo_returnsTwo() {
      assertThat(VorbisUtil.iLog(2)).isEqualTo(2);
    }

    @Test
    @DisplayName("Given a value of 3")
    public void iLog_whenThree_returnsTwo() {
      assertThat(VorbisUtil.iLog(3)).isEqualTo(2);
    }

    @Test
    @DisplayName("Given a value of 4")
    public void iLog_whenFour_returnsThree() {
      assertThat(VorbisUtil.iLog(4)).isEqualTo(3);
    }

    @Test
    @DisplayName("Given a value of 5")
    public void iLog_whenFive_returnsThree() {
      assertThat(VorbisUtil.iLog(5)).isEqualTo(3);
    }

    @Test
    @DisplayName("Given a value of 8")
    public void iLog_whenEight_returnsFour() {
      assertThat(VorbisUtil.iLog(8)).isEqualTo(4);
    }

    @Test
    @DisplayName("Given a negative value")
    public void iLog_whenNegative_returnsZero() {
      assertThat(VorbisUtil.iLog(-1)).isEqualTo(0);
      assertThat(VorbisUtil.iLog(-122)).isEqualTo(0);
    }
  }

  @Nested
  @DisplayName("When parsing the identification header")
  public class IdentificationHeaderTests {

    @Test
    @DisplayName("Parses the identification header correctly")
    public void readVorbisIdentificationHeader_parsesCorrectly() throws Exception {
      byte[] data = OggTestData.getIdentificationHeaderData();
      ParsableByteArray headerData = new ParsableByteArray(data, data.length);
      VorbisUtil.VorbisIdHeader header =
          VorbisUtil.readVorbisIdentificationHeader(headerData);

      assertThat(header.sampleRate).isEqualTo(22050);
      assertThat(header.version).isEqualTo(0);
      assertThat(header.framingFlag).isTrue();
      assertThat(header.channels).isEqualTo(2);
      assertThat(header.blockSize0).isEqualTo(512);
      assertThat(header.blockSize1).isEqualTo(1024);
      assertThat(header.bitrateMax).isEqualTo(-1);
      assertThat(header.bitrateMin).isEqualTo(-1);
      assertThat(header.bitrateNominal).isEqualTo(66666);
      assertThat(header.getApproximateBitrate()).isEqualTo(66666);
    }
  }

  @Nested
  @DisplayName("When parsing the comment header")
  public class CommentHeaderTests {

    @Test
    @DisplayName("Parses the comment header correctly")
    public void readVorbisCommentHeader_parsesCorrectly() throws ParserException {
      byte[] data = OggTestData.getCommentHeaderDataUTF8();
      ParsableByteArray headerData = new ParsableByteArray(data, data.length);
      VorbisUtil.CommentHeader header = VorbisUtil.readVorbisCommentHeader(headerData);

      assertThat(header.vendor)
          .isEqualTo("Xiph.Org libVorbis I 20120203 (Omnipresent)");
      assertThat(header.comments).hasLength(3);
      assertThat(header.comments[0]).isEqualTo("ALBUM=äö");
      assertThat(header.comments[1]).isEqualTo("TITLE=A sample song");
      assertThat(header.comments[2]).isEqualTo("ARTIST=Google");
    }
  }

  @Nested
  @DisplayName("When parsing the Vorbis modes")
  public class VorbisModeTests {

    @Test
    @DisplayName("Parses the Vorbis modes correctly")
    public void readVorbisModes_parsesCorrectly() throws ParserException {
      byte[] data = OggTestData.getSetupHeaderData();
      ParsableByteArray headerData = new ParsableByteArray(data, data.length);
      VorbisUtil.Mode[] modes = VorbisUtil.readVorbisModes(headerData, 2);

      assertThat(modes).hasLength(2);
      assertThat(modes[0].blockFlag).isFalse();
      assertThat(modes[0].mapping).isEqualTo(0);
      assertThat(modes[0].transformType).isEqualTo(0);
      assertThat(modes[0].windowType).isEqualTo(0);
      assertThat(modes[1].blockFlag).isTrue();
      assertThat(modes[1].mapping).isEqualTo(1);
      assertThat(modes[1].transformType).isEqualTo(0);
      assertThat(modes[1].windowType).isEqualTo(0);
    }
  }

  @Nested
  @DisplayName("When verifying Vorbis header capture patterns")
  public class VerifyVorbisHeaderCapturePatternTests {

    @Test
    @DisplayName("Returns true for a valid header capture pattern")
    public void verifyVorbisHeaderCapturePattern_withValidHeaderCapturePattern_returnsTrue()
        throws Exception {
      ParsableByteArray header =
          new ParsableByteArray(new byte[] {0x01, 'v', 'o', 'r', 'b', 'i', 's'});
      assertThat(verifyVorbisHeaderCapturePattern(0x01, header, false)).isTrue();
    }

    @Test
    @DisplayName("Throws a ParserException for an invalid header number")
    public void verifyVorbisHeaderCapturePattern_withInvalidHeaderNumber_throwsParserException() {
      ParsableByteArray header =
          new ParsableByteArray(new byte[] {0x01, 'v', 'o', 'r', 'b', 'i', 's'});
      try {
        VorbisUtil.verifyVorbisHeaderCapturePattern(0x99, header, false);
        fail();
      } catch (ParserException e) {
        assertThat(e.getMessage()).isEqualTo("expected header type 99");
      }
    }

    @Test
    @DisplayName("Returns false for an invalid header capture pattern in quiet mode")
    public void verifyVorbisHeaderCapturePattern_withInvalidHeaderPattern_inQuietMode_returnsFalse()
        throws ParserException {
      ParsableByteArray header =
          new ParsableByteArray(new byte[] {0x01, 'x', 'v', 'o', 'r', 'b', 'i', 's'});
      assertThat(verifyVorbisHeaderCapturePattern(0x01, header, true)).isFalse();
    }

    @Test
    @DisplayName("Throws a ParserException for an invalid header capture pattern")
    public void
        verifyVorbisHeaderCapturePattern_withInvalidHeaderPattern_throwsParserException() {
      ParsableByteArray header =
          new ParsableByteArray(new byte[] {0x01, 'x', 'v', 'o', 'r', 'b', 'i', 's'});
      try {
        VorbisUtil.verifyVorbisHeaderCapturePattern(0x01, header, false);
        fail();
      } catch (ParserException e) {
        assertThat(e.getMessage()).isEqualTo("expected characters 'vorbis'");
      }
    }
  }
} 