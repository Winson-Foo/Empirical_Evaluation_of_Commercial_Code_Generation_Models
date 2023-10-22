package com.google.android.exoplayer2.playbacktests.gts;

import android.media.MediaCodecInfo.AudioCapabilities;
import android.media.MediaCodecInfo.CodecCapabilities;
import android.media.MediaCodecInfo.CodecProfileLevel;
import android.media.MediaCodecInfo.VideoCapabilities;

import androidx.test.ext.junit.runners.AndroidJUnit4;

import com.google.android.exoplayer2.mediacodec.MediaCodecInfo;
import com.google.android.exoplayer2.mediacodec.MediaCodecUtil;
import com.google.android.exoplayer2.mediacodec.MediaCodecUtil.DecoderQueryException;
import com.google.android.exoplayer2.testutil.MetricsLogger;
import com.google.android.exoplayer2.util.Assertions;
import com.google.android.exoplayer2.util.MimeTypes;
import com.google.android.exoplayer2.util.Util;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.Arrays;
import java.util.List;

@RunWith(AndroidJUnit4.class)
public class EnumerateDecodersTest {

    private static final String TAG = "EnumerateDecodersTest";

    private static final boolean SECURE = true;
    private static final boolean TUNNELING = true;

    private MetricsLogger metricsLogger;

    @Before
    public void setUp() {
        metricsLogger = MetricsLogger.Factory.createDefault(TAG);
    }

    @Test
    public void testEnumerateDecoders() throws Exception {
        enumerateDecoders(MimeTypes.VIDEO_H263);
        enumerateDecoders(MimeTypes.VIDEO_H264);
        enumerateDecoders(MimeTypes.VIDEO_H265);
        enumerateDecoders(MimeTypes.VIDEO_VP8);
        enumerateDecoders(MimeTypes.VIDEO_VP9);
        enumerateDecoders(MimeTypes.VIDEO_MP4V);
        enumerateDecoders(MimeTypes.VIDEO_MPEG);
        enumerateDecoders(MimeTypes.VIDEO_MPEG2);
        enumerateDecoders(MimeTypes.VIDEO_VC1);
        enumerateDecoders(MimeTypes.AUDIO_AAC);
        enumerateDecoders(MimeTypes.AUDIO_MPEG_L1);
        enumerateDecoders(MimeTypes.AUDIO_MPEG_L2);
        enumerateDecoders(MimeTypes.AUDIO_MPEG);
        enumerateDecoders(MimeTypes.AUDIO_RAW);
        enumerateDecoders(MimeTypes.AUDIO_ALAW);
        enumerateDecoders(MimeTypes.AUDIO_MLAW);
        enumerateDecoders(MimeTypes.AUDIO_AC3);
        enumerateDecoders(MimeTypes.AUDIO_E_AC3);
        enumerateDecoders(MimeTypes.AUDIO_E_AC3_JOC);
        enumerateDecoders(MimeTypes.AUDIO_TRUEHD);
        enumerateDecoders(MimeTypes.AUDIO_DTS);
        enumerateDecoders(MimeTypes.AUDIO_DTS_HD);
        enumerateDecoders(MimeTypes.AUDIO_DTS_EXPRESS);
        enumerateDecoders(MimeTypes.AUDIO_VORBIS);
        enumerateDecoders(MimeTypes.AUDIO_OPUS);
        enumerateDecoders(MimeTypes.AUDIO_AMR_NB);
        enumerateDecoders(MimeTypes.AUDIO_AMR_WB);
        enumerateDecoders(MimeTypes.AUDIO_FLAC);
        enumerateDecoders(MimeTypes.AUDIO_ALAC);
        enumerateDecoders(MimeTypes.AUDIO_MSGSM);
    }

    private void enumerateDecoders(String mimeType) throws DecoderQueryException {
        logDecoderInfos(mimeType, !SECURE, !TUNNELING);
        logDecoderInfos(mimeType, SECURE, !TUNNELING);
        logDecoderInfos(mimeType, !SECURE, TUNNELING);
        logDecoderInfos(mimeType, SECURE, TUNNELING);
    }

    private void logDecoderInfos(String mimeType, boolean secure, boolean tunneling)
            throws DecoderQueryException {
        List<MediaCodecInfo> mediaCodecInfos = MediaCodecUtil.getDecoderInfos(mimeType, secure, tunneling);
        for (MediaCodecInfo mediaCodecInfo : mediaCodecInfos) {
            CodecCapabilities capabilities = Assertions.checkNotNull(mediaCodecInfo.capabilities);
            metricsLogger.logMetric(getCapabilitiesKey(mediaCodecInfo.name), codecCapabilitiesToString(mimeType, capabilities));
        }
    }

    private String getCapabilitiesKey(String name) {
        return "capabilities_" + name;
    }

    private static String codecCapabilitiesToString(String mimeType, CodecCapabilities codecCapabilities) {
        StringBuilder builder = new StringBuilder();
        builder.append("[requestedMimeType=").append(mimeType);
        if (Util.SDK_INT >= 21) {
            builder.append(", mimeType=").append(codecCapabilities.getMimeType());
        }
        builder.append(", profileLevels=");
        appendProfileLevels(codecCapabilities.profileLevels, builder);
        if (Util.SDK_INT >= 23) {
            builder.append(", maxSupportedInstances=").append(codecCapabilities.getMaxSupportedInstances());
        }
        if (Util.SDK_INT >= 21) {
            if (MimeTypes.isVideo(mimeType)) {
                builder.append(", videoCapabilities=");
                appendVideoCapabilities(codecCapabilities.getVideoCapabilities(), builder);
                builder.append(", colorFormats=").append(Arrays.toString(codecCapabilities.colorFormats));
            } else if (MimeTypes.isAudio(mimeType)) {
                builder.append(", audioCapabilities=");
                appendAudioCapabilities(codecCapabilities.getAudioCapabilities(), builder);
            }
        }
        if (Util.SDK_INT >= 19 && MimeTypes.isVideo(mimeType) && codecCapabilities.isFeatureSupported(CodecCapabilities.FEATURE_AdaptivePlayback)) {
            builder.append(", FEATURE_AdaptivePlayback");
        }
        if (Util.SDK_INT >= 21 && MimeTypes.isVideo(mimeType) && codecCapabilities.isFeatureSupported(CodecCapabilities.FEATURE_SecurePlayback)) {
            builder.append(", FEATURE_SecurePlayback");
        }
        if (Util.SDK_INT >= 26 && MimeTypes.isVideo(mimeType) && codecCapabilities.isFeatureSupported(CodecCapabilities.FEATURE_PartialFrame)) {
            builder.append(", FEATURE_PartialFrame");
        }
        if (Util.SDK_INT >= 21 && (MimeTypes.isVideo(mimeType) || MimeTypes.isAudio(mimeType)) && codecCapabilities.isFeatureSupported(CodecCapabilities.FEATURE_TunneledPlayback)) {
            builder.append(", FEATURE_TunneledPlayback");
        }
        builder.append(']');
        return builder.toString();
    }

    private static void appendAudioCapabilities(AudioCapabilities capabilities, StringBuilder builder) {
        builder
                .append("[bitrateRange=").append(capabilities.getBitrateRange())
                .append(", maxInputChannelCount=").append(capabilities.getMaxInputChannelCount())
                .append(", supportedSampleRateRanges=").append(Arrays.toString(capabilities.getSupportedSampleRateRanges()))
                .append(']');
    }

    private static void appendVideoCapabilities(VideoCapabilities capabilities, StringBuilder builder) {
        builder
                .append("[bitrateRange=").append(capabilities.getBitrateRange())
                .append(", heightAlignment=").append(capabilities.getHeightAlignment())
                .append(", widthAlignment=").append(capabilities.getWidthAlignment())
                .append(", supportedWidths=").append(capabilities.getSupportedWidths())
                .append(", supportedHeights=").append(capabilities.getSupportedHeights())
                .append(", supportedFrameRates=").append(capabilities.getSupportedFrameRates())
                .append(']');
    }

    private static void appendProfileLevels(CodecProfileLevel[] profileLevels, StringBuilder builder) {
        builder.append('[');
        int count = profileLevels.length;
        for (int i = 0; i < count; i++) {
            CodecProfileLevel profileLevel = profileLevels[i];
            if (i != 0) {
                builder.append(", ");
            }
            builder.append("[profile=").append(profileLevel.profile).append(", level=").append(profileLevel.level).append(']');
        }
        builder.append(']');
    }
} 