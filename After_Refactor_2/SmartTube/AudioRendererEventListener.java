package com.google.android.exoplayer2.audio;

import android.os.Handler;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import com.google.android.exoplayer2.Format;
import com.google.android.exoplayer2.decoder.DecoderCounters;

public interface AudioRendererEventListener {

  void onAudioEnabled(@NonNull DecoderCounters counters);

  void onAudioSessionId(int audioSessionId);

  void onAudioDecoderInitialized(
      String decoderName, long initializedTimestampMs, long initializationDurationMs);

  void onAudioInputFormatChanged(@NonNull Format format);

  void onAudioSinkUnderrun(int bufferSize, long bufferSizeMs, long elapsedSinceLastFeedMs);

  void onAudioDisabled(@NonNull DecoderCounters counters);

  @SuppressWarnings("NullableProblems")
  final class EventDispatcher {

    @NonNull private final Handler handler;
    @Nullable private final AudioRendererEventListener listener;

    @SuppressWarnings("NullableProblems")
    @AllArgsConstructor
    public EventDispatcher(@NonNull Handler handler, @Nullable AudioRendererEventListener listener) {
      this.handler = handler;
      this.listener = listener;
    }

    public void audioEnabled(@NonNull DecoderCounters counters) {
      handler.post(() -> listener.onAudioEnabled(counters));
    }

    public void audioSessionId(int audioSessionId) {
      handler.post(() -> listener.onAudioSessionId(audioSessionId));
    }

    public void audioDecoderInitialized(String decoderName, long initializedTimestampMs,
        long initializationDurationMs) {
      handler.post(
          () ->
              listener.onAudioDecoderInitialized(
                  decoderName, initializedTimestampMs, initializationDurationMs));
    }

    public void audioInputFormatChanged(@NonNull Format format) {
      handler.post(() -> listener.onAudioInputFormatChanged(format));
    }

    public void audioSinkUnderrun(int bufferSize, long bufferSizeMs, long elapsedSinceLastFeedMs) {
      handler.post(
          () -> listener.onAudioSinkUnderrun(bufferSize, bufferSizeMs, elapsedSinceLastFeedMs));
    }

    public void audioDisabled(@NonNull DecoderCounters counters) {
      counters.ensureUpdated();
      handler.post(() -> listener.onAudioDisabled(counters));
    }
  }
} 