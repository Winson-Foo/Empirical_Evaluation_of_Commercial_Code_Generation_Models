package com.google.android.exoplayer2;

import static com.google.android.exoplayer2.C.WIDEVINE_UUID;
import static com.google.android.exoplayer2.util.MimeTypes.AUDIO_AAC;
import static com.google.android.exoplayer2.util.MimeTypes.AUDIO_OPUS;
import static com.google.android.exoplayer2.util.MimeTypes.VIDEO_H264;
import static com.google.common.truth.Truth.assertThat;

import android.os.Parcel;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.android.exoplayer2.drm.DrmInitData;
import com.google.android.exoplayer2.metadata.Metadata;
import com.google.android.exoplayer2.metadata.id3.TextInformationFrame;
import com.google.android.exoplayer2.testutil.TestUtil;
import com.google.android.exoplayer2.util.MimeTypes;
import com.google.android.exoplayer2.video.ColorInfo;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Unit test for {@link Format}. */
@RunWith(AndroidJUnit4.class)
public final class FormatTest {

  private static final String ID = "id";
  private static final String LABEL = "label";
  private static final int SELECTION_FLAG = C.SELECTION_FLAG_DEFAULT;
  private static final int ROLE_FLAG = C.ROLE_FLAG_MAIN;
  private static final int BITRATE = 1024;
  private static final String CODEC = "codec";
  private static final int MAX_INPUT_SIZE = 2048;
  private static final int WIDTH = 1920;
  private static final int HEIGHT = 1080;
  private static final float FRAME_RATE = 24f;
  private static final int ROTATION_DEGREES = 90;
  private static final float PIXEL_WIDTH_HEIGHT_RATIO = 2f;
  private static final int CHANNEL_COUNT = 6;
  private static final int SAMPLE_RATE = 44100;
  private static final int ENCODING = C.ENCODING_PCM_24BIT;
  private static final int ENCODER_DELAY = 1001;
  private static final int ENCODER_PADDING = 1002;
  private static final String LANGUAGE = "language";
  private static final int ACCESSIBILITY_CHANNEL = Format.NO_VALUE;
  private static final String PROJECTION_DATA = "projectionData";
  private static final int STEREO_MODE = C.STEREO_MODE_TOP_BOTTOM;

  private static final List<byte[]> INIT_DATA = Collections.unmodifiableList(
      new ArrayList<byte[]>() {
        {
          add(new byte[] {1, 2, 3});
          add(new byte[] {4, 5, 6});
        }
      });

  @Test
  public void testParcelable() {
    Format formatToParcel = createTestFormat();

    Parcel parcel = Parcel.obtain();
    formatToParcel.writeToParcel(parcel, 0);
    parcel.setDataPosition(0);

    Format formatFromParcel = Format.CREATOR.createFromParcel(parcel);
    assertThat(formatFromParcel).isEqualTo(formatToParcel);

    parcel.recycle();
  }

  private Format createTestFormat() {
    DrmInitData.SchemeData drmData1 = createTestSchemeData(WIDEVINE_UUID, VIDEO_H264, 1);
    DrmInitData.SchemeData drmData2 = createTestSchemeData(C.UUID_NIL, MimeTypes.VIDEO_WEBM, 2);
    DrmInitData drmInitData = new DrmInitData(drmData1, drmData2);
    Metadata metadata = createTestMetadata();
    ColorInfo colorInfo = createTestColorInfo();

    return new FormatBuilder()
        .setId(ID)
        .setLabel(LABEL)
        .setSelectionFlag(SELECTION_FLAG)
        .setRoleFlag(ROLE_FLAG)
        .setBitrate(BITRATE)
        .setCodec(CODEC)
        .setContainerMimeType(MimeTypes.VIDEO_MP4)
        .setSampleMimeType(MimeTypes.VIDEO_H264)
        .setMaxInputSize(MAX_INPUT_SIZE)
        .setInitializationData(INIT_DATA)
        .setDrmInitData(drmInitData)
        .setOffsetToAbsoluteSeekPosition(Format.OFFSET_SAMPLE_RELATIVE)
        .setWidth(WIDTH)
        .setHeight(HEIGHT)
        .setFrameRate(FRAME_RATE)
        .setRotationDegrees(ROTATION_DEGREES)
        .setPixelWidthHeightRatio(PIXEL_WIDTH_HEIGHT_RATIO)
        .setProjectionData(PROJECTION_DATA)
        .setStereoMode(STEREO_MODE)
        .setColorInfo(colorInfo)
        .setChannelCount(CHANNEL_COUNT)
        .setSampleRate(SAMPLE_RATE)
        .setPcmEncoding(ENCODING)
        .setEncoderDelay(ENCODER_DELAY)
        .setEncoderPadding(ENCODER_PADDING)
        .setLanguage(LANGUAGE)
        .setAccessibilityChannel(ACCESSIBILITY_CHANNEL)
        .setMetadata(metadata)
        .build();
  }

  private DrmInitData.SchemeData createTestSchemeData(
      UUID uuid, String mimeType, int seed) {
    byte[] data = TestUtil.buildTestData(128, seed);
    return new DrmInitData.SchemeData(uuid, mimeType, data);
  }

  private Metadata createTestMetadata() {
    return new Metadata(
        new TextInformationFrame("id1", "description1", "value1"),
        new TextInformationFrame("id2", "description2", "value2"));
  }

  private ColorInfo createTestColorInfo() {
    return new ColorInfo(
        C.COLOR_SPACE_BT709,
        C.COLOR_RANGE_LIMITED,
        C.COLOR_TRANSFER_SDR,
        new byte[] {1, 2, 3, 4, 5, 6, 7});
  }

  private static class FormatBuilder {
    private final Format.Builder builder = new Format.Builder();

    public FormatBuilder setId(String id) {
      builder.setId(id);
      return this;
    }

    public FormatBuilder setLabel(String label){
      builder.setLabel(label);
      return this;
    }

    public FormatBuilder setSelectionFlag(int selectionFlag) {
      builder.setSelectionFlag(selectionFlag);
      return this;
    }

    public FormatBuilder setRoleFlag(int roleFlag) {
      builder.setRoleFlag(roleFlag);
      return this;
    }

    public FormatBuilder setBitrate(int bitrate) {
      builder.setBitrate(bitrate);
      return this;
    }

    public FormatBuilder setCodec(String codec) {
      builder.setCodec(codec);
      return this;
    }

    public FormatBuilder setContainerMimeType(String containerMimeType) {
      builder.setContainerMimeType(containerMimeType);
      return this;
    }

    public FormatBuilder setSampleMimeType(String sampleMimeType) {
      builder.setSampleMimeType(sampleMimeType);
      return this;
    }

    public FormatBuilder setMaxInputSize(int maxInputSize) {
      builder.setMaxInputSize(maxInputSize);
      return this;
    }

    public FormatBuilder setInitializationData(List<byte[]> initializationData) {
      builder.setInitializationData(initializationData);
      return this;
    }

    public FormatBuilder setDrmInitData(DrmInitData drmInitData) {
      builder.setDrmInitData(drmInitData);
      return this;
    }

    public FormatBuilder setOffsetToAbsoluteSeekPosition(int offset) {
      builder.setOffsetToAbsoluteSeekPosition(offset);
      return this;
    }

    public FormatBuilder setWidth(int width) {
      builder.setWidth(width);
      return this;
    }

    public FormatBuilder setHeight(int height){
      builder.setHeight(height);
      return this;
    }

    public FormatBuilder setFrameRate(float frameRate) {
      builder.setFrameRate(frameRate);
      return this;
    }

    public FormatBuilder setRotationDegrees(int rotationDegrees) {
      builder.setRotationDegrees(rotationDegrees);
      return this;
    }

    public FormatBuilder setPixelWidthHeightRatio(float pixelWidthHeightRatio) {
      builder.setPixelWidthHeightRatio(pixelWidthHeightRatio);
      return this;
    }

    public FormatBuilder setProjectionData(String projectionData) {
      builder.setProjectionData(projectionData);
      return this;
    }

    public FormatBuilder setStereoMode(int stereoMode) {
      builder.setStereoMode(stereoMode);
      return this;
    }

    public FormatBuilder setColorInfo(ColorInfo colorInfo) {
      builder.setColorInfo(colorInfo);
      return this;
    }

    public FormatBuilder setChannelCount(int channelCount) {
      builder.setChannelCount(channelCount);
      return this;
    }

    public FormatBuilder setSampleRate(int sampleRate) {
      builder.setSampleRate(sampleRate);
      return this;
    }

    public FormatBuilder setPcmEncoding(int pcmEncoding) {
      builder.setPcmEncoding(pcmEncoding);
      return this;
    }

    public FormatBuilder setEncoderDelay(int encoderDelay) {
      builder.setEncoderDelay(encoderDelay);
      return this;
    }

    public FormatBuilder setEncoderPadding(int encoderPadding) {
      builder.setEncoderPadding(encoderPadding);
      return this;
    }

    public FormatBuilder setLanguage(String language) {
      builder.setLanguage(language);
      return this;
    }

    public FormatBuilder setAccessibilityChannel(int accessibilityChannel) {
      builder.setAccessibilityChannel(accessibilityChannel);
      return this;
    }

    public FormatBuilder setMetadata(Metadata metadata) {
      builder.setMetadata(metadata);
      return this;
    }

    public Format build() {
      return builder.build();
    }
  }
} 