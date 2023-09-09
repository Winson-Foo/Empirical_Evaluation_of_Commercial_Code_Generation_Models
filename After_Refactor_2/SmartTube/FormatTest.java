package com.google.android.exoplayer2;

import android.os.Parcel;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.android.exoplayer2.drm.DrmInitData;
import com.google.android.exoplayer2.metadata.Metadata;
import com.google.android.exoplayer2.metadata.id3.TextInformationFrame;
import com.google.android.exoplayer2.testutil.TestUtil;
import com.google.android.exoplayer2.video.ColorInfo;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Unit test for {@link Format}. */
@RunWith(AndroidJUnit4.class)
public final class FormatTest {

  private static final List<byte[]> INIT_DATA;
  static {
    byte[] initData1 = new byte[] {1, 2, 3};
    byte[] initData2 = new byte[] {4, 5, 6};
    List<byte[]> initDataList = new ArrayList<>();
    initDataList.add(initData1);
    initDataList.add(initData2);
    INIT_DATA = Collections.unmodifiableList(initDataList);
  }

  @Test
  public void testParcelable() {
    // Define test data.
    DrmInitData.SchemeData drmData1 =
        new DrmInitData.SchemeData(
            C.WIDEVINE_UUID,
            MimeTypes.VIDEO_MP4,
            TestUtil.buildTestData(128, 1 /* data seed */));
    DrmInitData.SchemeData drmData2 =
        new DrmInitData.SchemeData(
            C.UUID_NIL,
            MimeTypes.VIDEO_WEBM,
            TestUtil.buildTestData(128, 1 /* data seed */));
    DrmInitData drmInitData = new DrmInitData(drmData1, drmData2);
    byte[] projectionData = new byte[] {1, 2, 3};
    Metadata metadata =
        new Metadata(
            new TextInformationFrame("id1", "description1", "value1"),
            new TextInformationFrame("id2", "description2", "value2"));
    ColorInfo colorInfo =
        new ColorInfo(
            C.COLOR_SPACE_BT709,
            C.COLOR_RANGE_LIMITED,
            C.COLOR_TRANSFER_SDR,
            new byte[] {1, 2, 3, 4, 5, 6, 7});

    // Build the format using a builder.
    Format formatToParcel =
        new Format.Builder()
            .setId("id")
            .setLabel("label")
            .setSelectionFlags(C.SELECTION_FLAG_DEFAULT)
            .setRoleFlags(C.ROLE_FLAG_MAIN)
            .setBitrate(1024)
            .setCodec("codec")
            .setMetadata(metadata)
            .setContainerMimeType(MimeTypes.VIDEO_MP4)
            .setSampleMimeType(MimeTypes.VIDEO_H264)
            .setMaxInputSize(2048)
            .setInitializationData(INIT_DATA)
            .setDrmInitData(drmInitData)
            .setOffsetSampleIndex(Format.OFFSET_SAMPLE_RELATIVE)
            .setWidth(1920)
            .setHeight(1080)
            .setFrameRate(24)
            .setRotationDegrees(90)
            .setPixelWidthHeightRatio(2)
            .setProjectionData(projectionData)
            .setStereoMode(C.STEREO_MODE_TOP_BOTTOM)
            .setColorInfo(colorInfo)
            .setChannelCount(6)
            .setSampleRate(44100)
            .setPcmEncoding(C.ENCODING_PCM_24BIT)
            .setEncoderDelay(1001)
            .setEncoderPadding(1002)
            .setLanguage("language")
            .setAccessibilityChannel(Format.NO_VALUE)
            .build();

    // Test that the format can be parceled and unparceled without losing any data.
    Parcel parcel = Parcel.obtain();
    formatToParcel.writeToParcel(parcel, 0);
    parcel.setDataPosition(0);

    Format formatFromParcel = Format.CREATOR.createFromParcel(parcel);
    assertThat(formatFromParcel).isEqualTo(formatToParcel);

    parcel.recycle();
  }
} 