public interface AudioRendererEventListener {

    default void onAudioEnabled(DecoderCounters counters) {}

    default void onAudioSessionId(int audioSessionId) {}

    default void onAudioDecoderInitialized(String decoderName, long timestampMs, long durationMs) {}

    default void onAudioInputFormatChanged(Format format) {}

    default void onAudioSinkUnderrun(int bufferSize, long bufferSizeMs, long elapsedMs) {}

    default void onAudioDisabled(DecoderCounters counters) {}

} 