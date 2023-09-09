package com.google.android.exoplayer2.drm;

import androidx.annotation.NonNull;

import com.google.android.exoplayer2.C;

import java.util.Map;

/**
 * Utility methods for Widevine.
 */
public final class WidevineUtil {

    /**
     * Constants used by Widevine.
     */
    public interface Constants {
        String LICENSE_DURATION_REMAINING = "LicenseDurationRemaining";
        String PLAYBACK_DURATION_REMAINING = "PlaybackDurationRemaining";
    }

    private WidevineUtil() {}

    /**
     * Returns license and playback durations remaining in seconds.
     *
     * @param drmSession The drm session to query.
     * @return A {@link WidevineDurationRemaining} object consisting of the remaining license and playback
     * durations in seconds.
     */
    public static WidevineDurationRemaining getDurationRemaining(@NonNull DrmSession<?> drmSession) {
        Map<String, String> keyStatus = drmSession.queryKeyStatus();
        long licenseDurationRemainingSec = extractDurationFromKeyStatus(keyStatus, Constants.LICENSE_DURATION_REMAINING);
        long playbackDurationRemainingSec = extractDurationFromKeyStatus(keyStatus, Constants.PLAYBACK_DURATION_REMAINING);
        return new WidevineDurationRemaining(licenseDurationRemainingSec, playbackDurationRemainingSec);
    }

    private static long extractDurationFromKeyStatus(@NonNull Map<String, String> keyStatus, @NonNull String property) {
        String value = keyStatus.get(property);
        if (value != null) {
            try {
                return Long.parseLong(value);
            } catch (NumberFormatException e) {
                // ignore
            }
        }
        return C.TIME_UNSET;
    }

    /**
     * A helper class representing the remaining durations in seconds for a Widevine DRM session.
     */
    public static final class WidevineDurationRemaining {

        private final long licenseDurationRemainingSec;
        private final long playbackDurationRemainingSec;

        public WidevineDurationRemaining(long licenseDurationRemainingSec, long playbackDurationRemainingSec) {
            this.licenseDurationRemainingSec = licenseDurationRemainingSec;
            this.playbackDurationRemainingSec = playbackDurationRemainingSec;
        }

        /**
         * Returns the remaining license duration in seconds.
         *
         * @return The remaining license duration in seconds.
         */
        public long getLicenseDurationRemainingSec() {
            return licenseDurationRemainingSec;
        }

        /**
         * Returns the remaining playback duration in seconds.
         *
         * @return The remaining playback duration in seconds.
         */
        public long getPlaybackDurationRemainingSec() {
            return playbackDurationRemainingSec;
        }
    }

}

