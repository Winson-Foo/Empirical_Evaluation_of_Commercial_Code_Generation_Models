package com.google.android.exoplayer2.audio;

import com.google.android.exoplayer2.Format;
import com.google.android.exoplayer2.decoder.DecoderCounters;

public interface AudioRendererEventListener {
    void onAudioEnabled(DecoderCounters decoderCounters);
    void onAudioSessionId(int audioSessionId);
    void onAudioDecoderInitialized(
            String decoderName,
            long initializedTimestampMs,
            long initializationDurationMs);
    void onAudioInputFormatChanged(Format format);
    void onAudioSinkUnderrun(
            int bufferSize,
            long bufferSizeMs,
            long elapsedSinceLastFeedMs);
    void onAudioDisabled(DecoderCounters decoderCounters);
}