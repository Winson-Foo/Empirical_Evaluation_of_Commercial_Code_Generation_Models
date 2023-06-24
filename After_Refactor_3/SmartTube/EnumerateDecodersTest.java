package com.google.android.exoplayer2.playbacktests.gts;

import android.media.MediaCodecInfo;
import android.media.MediaCodecInfo.AudioCapabilities;
import android.media.MediaCodecInfo.CodecCapabilities;
import android.media.MediaCodecInfo.CodecProfileLevel;
import android.media.MediaCodecInfo.VideoCapabilities;

import androidx.test.ext.junit.runners.AndroidJUnit4;

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

    private MetricsLogger metricsLogger;

    private static final List<String> MIME_TYPES = Arrays.asList(
            MimeTypes.VIDEO_H263,
            MimeTypes.VIDEO_H264,
            MimeTypes.VIDEO_H265,
            MimeTypes.VIDEO_VP8,
            MimeTypes.VIDEO_VP9,
            MimeTypes.VIDEO_MP4V,
            MimeTypes.VIDEO_MPEG,
            MimeTypes.VIDEO_MPEG2,
            MimeTypes.VIDEO_VC1,
            MimeTypes.AUDIO_AAC,
            MimeTypes.AUDIO_MPEG_L1,
            MimeTypes.AUDIO_MPEG_L2,
            MimeTypes.AUDIO_MPEG,
            MimeTypes.AUDIO_RAW,
            MimeTypes.AUDIO_ALAW,
            MimeTypes.AUDIO_MLAW,
            MimeTypes.AUDIO_AC3,
            MimeTypes.AUDIO_E_AC3,
            MimeTypes.AUDIO_E_AC3_JOC,
            MimeTypes.AUDIO_TRUEHD,
            MimeTypes.AUDIO_DTS,
            MimeTypes.AUDIO_DTS_HD,
            MimeTypes.AUDIO_DTS_EXPRESS,
            MimeTypes.AUDIO_VORBIS,
            MimeTypes.AUDIO_OPUS,
            MimeTypes.AUDIO_AMR_NB,
            MimeTypes.AUDIO_AMR_WB,
            MimeTypes.AUDIO_FLAC,
            MimeTypes.AUDIO_ALAC,
            MimeTypes.AUDIO_MSGSM
    );

    @Before
    public void setUp() {
        metricsLogger = MetricsLogger.Factory.createDefault(TAG);
    }

    @Test
    public void testEnumerateDecoders() throws Exception {
        for (String mimeType : MIME_TYPES) {
            enumerateDecoders(mimeType);
        }
    }

    private void enumerateDecoders(String mimeType) throws DecoderQueryException {
        logDecoderInfos(mimeType, false, false);
        logDecoderInfos(mimeType, true, false);
        logDecoderInfos(mimeType, false, true);
        logDecoderInfos(mimeType, true, true);
    }

    private void logDecoderInfos(String mimeType, boolean secure, boolean tunneling)
            throws DecoderQueryException {
        List<MediaCodecInfo> mediaCodecInfos =
                MediaCodecUtil.getDecoderInfos(mimeType, secure, tunneling);
        for (MediaCodecInfo mediaCodecInfo : mediaCodecInfos) {
            CodecCapabilities capabilities = Assertions.checkNotNull(mediaCodecInfo.capabilities);
            metricsLogger.logMetric(
                    "capabilities_" + mediaCodecInfo.name,
                    codecCapabilitiesToString(mimeType, capabilities)
            );
        }
    }

    private static String codecCapabilitiesToString(
            String requestedMimeType, CodecCapabilities codecCapabilities) {
        StringBuilder result = new StringBuilder("[requestedMimeType=").append(requestedMimeType);
        if (Util.SDK_INT >= 21) {
            result.append(", mimeType=").append(codecCapabilities.getMimeType());
        }
        result.append(", profileLevels=");
        appendProfileLevels(codecCapabilities.profileLevels, result);
        if (Util.SDK_INT >= 23) {
            result.append(", maxSupportedInstances=").append(
                    codecCapabilities.getMaxSupportedInstances());
        }
        if (Util.SDK_INT >= 21 && MimeTypes.isVideo(requestedMimeType)) {
            result.append(", videoCapabilities=");
            appendVideoCapabilities(codecCapabilities.getVideoCapabilities(), result);
            result.append(", colorFormats=").append(Arrays.toString(codecCapabilities.colorFormats));
        } else if (Util.SDK_INT >= 21 && MimeTypes.isAudio(requestedMimeType)) {
            result.append(", audioCapabilities=");
            appendAudioCapabilities(codecCapabilities.getAudioCapabilities(), result);
        }
        if (Util.SDK_INT >= 19 && MimeTypes.isVideo(requestedMimeType)
                && codecCapabilities.isFeatureSupported(
                CodecCapabilities.FEATURE_AdaptivePlayback)) {
            result.append(", FEATURE_AdaptivePlayback");
        }
        if (Util.SDK_INT >= 21 && MimeTypes.isVideo(requestedMimeType)
                && codecCapabilities.isFeatureSupported(
                CodecCapabilities.FEATURE_SecurePlayback)) {
            result.append(", FEATURE_SecurePlayback");
        }
        if (Util.SDK_INT >= 26 && MimeTypes.isVideo(requestedMimeType)
                && codecCapabilities.isFeatureSupported(
                CodecCapabilities.FEATURE_PartialFrame)) {
            result.append(", FEATURE_PartialFrame");
        }
        if (Util.SDK_INT >= 21 && (MimeTypes.isVideo(requestedMimeType) || MimeTypes.isAudio(requestedMimeType))
                && codecCapabilities.isFeatureSupported(
                CodecCapabilities.FEATURE_TunneledPlayback)) {
            result.append(", FEATURE_TunneledPlayback");
        }
        result.append(']');
        return result.toString();
    }

    private static void appendAudioCapabilities(
            AudioCapabilities audioCapabilities, StringBuilder result) {
        result
                .append("[bitrateRange=")
                .append(audioCapabilities.getBitrateRange())
                .append(", maxInputChannelCount=")
                .append(audioCapabilities.getMaxInputChannelCount())
                .append(", supportedSampleRateRanges=")
                .append(Arrays.toString(audioCapabilities.getSupportedSampleRateRanges()))
                .append(']');
    }

    private static void appendVideoCapabilities(
            VideoCapabilities videoCapabilities, StringBuilder result) {
        result
                .append("[bitrateRange=")
                .append(videoCapabilities.getBitrateRange())
                .append(", heightAlignment=")
                .append(videoCapabilities.getHeightAlignment())
                .append(", widthAlignment=")
                .append(videoCapabilities.getWidthAlignment())
                .append(", supportedWidths=")
                .append(videoCapabilities.getSupportedWidths())
                .append(", supportedHeights=")
                .append(videoCapabilities.getSupportedHeights())
                .append(", supportedFrameRates=")
                .append(videoCapabilities.getSupportedFrameRates())
                .append(']');
    }

    private static void appendProfileLevels(CodecProfileLevel[] profileLevels, StringBuilder result) {
        result.append('[');
        int count = profileLevels.length;
        for (int i = 0; i < count; i++) {
            CodecProfileLevel profileLevel = profileLevels[i];
            if (i != 0) {
                result.append(", ");
            }
            result
                    .append("[profile=")
                    .append(profileLevel.profile)
                    .append(", level=")
                    .append(profileLevel.level)
                    .append(']');
        }
        result.append(']');
    }
} 