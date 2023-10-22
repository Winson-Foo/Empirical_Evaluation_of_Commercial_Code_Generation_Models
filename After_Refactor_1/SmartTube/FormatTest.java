package com.google.android.exoplayer2;

import static com.google.android.exoplayer2.C.WIDEVINE_UUID;
import static com.google.android.exoplayer2.util.MimeTypes.VIDEO_H264;
import static com.google.android.exoplayer2.util.MimeTypes.VIDEO_MP4;
import static com.google.android.exoplayer2.util.MimeTypes.VIDEO_WEBM;
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

@RunWith(AndroidJUnit4.class)
public final class FormatTest {

  private static final byte[] INIT_DATA_1 = {1, 2, 3};
  private static final byte[] INIT_DATA_2 = {4, 5, 6};
  private static final List<byte[]> INIT_DATA = Collections.unmodifiableList(
      new ArrayList<byte[]>() {{
          add(INIT_DATA_1);
          add(INIT_DATA_2);
      }});
  private static final int BITRATE = 1024;
  private static final int MAX_INPUT_SIZE = 2048;
  private static final int WIDTH = 1920;
  private static final int HEIGHT = 1080;
  private static final float FRAME_RATE = 24f;
  private static final int ROTATION_DEGREES = 90;
  private static final float PIXEL_WIDTH_HEIGHT_RATIO = 2f;
  private static final int CHANNEL_COUNT = 6;
  private static final int SAMPLE_RATE = 44100;
  private static final int ENCODER_DELAY = 1001;
  private static final int ENCODER_PADDING = 1002;
  private static final String ID = "id";
  private static final String LABEL = "label";
  private static final int SELECTION_FLAG = C.SELECTION_FLAG_DEFAULT;
  private static final int ROLE_FLAG = C.ROLE_FLAG_MAIN;
  private static final String CODEC = "codec";
  private static final Metadata METADATA 
      = new Metadata(
          new TextInformationFrame("id1", "description1", "value1"),
          new TextInformationFrame("id2", "description2", "value2"));
  private static final String CONTAINER_MIME_TYPE = VIDEO_MP4;
  private static final String SAMPLE_MIME_TYPE = VIDEO_H264;
  private static final byte[] PROJECTION_DATA = {1, 2, 3};
  private static final int STEREO_MODE = C.STEREO_MODE_TOP_BOTTOM;
  private static final String LANGUAGE = "language";
  private static final int ACCESSIBILITY_CHANNEL = Format.NO_VALUE;
  private static final ColorInfo COLOR_INFO = new ColorInfo(
      C.COLOR_SPACE_BT709,
      C.COLOR_RANGE_LIMITED, 
      C.COLOR_TRANSFER_SDR, 
      new byte[] {1, 2, 3, 4, 5, 6, 7});

  @Test
  public void testParcelable() {
    Format formatToParcel = getTestFormat();
    Parcel parcel = Parcel.obtain();
    formatToParcel.writeToParcel(parcel, 0);
    parcel.setDataPosition(0);

    Format formatFromParcel = Format.CREATOR.createFromParcel(parcel);
    assertThat(formatFromParcel).isEqualTo(formatToParcel);

    parcel.recycle();
  }

  private Format getTestFormat() {
    DrmInitData.SchemeData drmData1 = new DrmInitData.SchemeData(WIDEVINE_UUID, VIDEO_MP4,
        TestUtil.buildTestData(128, 1 /* data seed */));
    DrmInitData.SchemeData drmData2 = new DrmInitData.SchemeData(C.UUID_NIL, VIDEO_WEBM,
        TestUtil.buildTestData(128, 1 /* data seed */));
    DrmInitData drmInitData = new DrmInitData(drmData1, drmData2);

    return new Format(
        ID,
        LABEL,
        SELECTION_FLAG,
        ROLE_FLAG,
        BITRATE,
        CODEC,
        METADATA,
        CONTAINER_MIME_TYPE,
        SAMPLE_MIME_TYPE,
        MAX_INPUT_SIZE,
        INIT_DATA,
        drmInitData,
        Format.OFFSET_SAMPLE_RELATIVE,
        WIDTH,
        HEIGHT,
        FRAME_RATE,
        ROTATION_DEGREES,
        PIXEL_WIDTH_HEIGHT_RATIO,
        PROJECTION_DATA,
        STEREO_MODE,
        COLOR_INFO,
        CHANNEL_COUNT,
        SAMPLE_RATE,
        C.ENCODING_PCM_24BIT,
        ENCODER_DELAY,
        ENCODER_PADDING,
        LANGUAGE,
        ACCESSIBILITY_CHANNEL);
  }

}

