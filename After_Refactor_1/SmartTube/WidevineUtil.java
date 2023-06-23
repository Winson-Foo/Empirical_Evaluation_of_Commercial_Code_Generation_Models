package com.google.android.exoplayer2.drm;

import android.util.Pair;

import androidx.annotation.Nullable;

import com.google.android.exoplayer2.C;

import java.util.Map;

public final class WidevineUtil {

    public static final String LICENSE_DURATION_REMAINING = "LicenseDurationRemaining";
    public static final String PLAYBACK_DURATION_REMAINING = "PlaybackDurationRemaining";

    private WidevineUtil() {
        //no instance
    }

    public static @Nullable Pair<Long, Long> getLicenseDurationRemaining(DrmSession<?> drmSession) {
        Map<String, String> keyStatus = drmSession.queryKeyStatus();
        if (keyStatus == null) {
            return null;
        }
        return new Pair<>(getDurationRemainingInSeconds(keyStatus, LICENSE_DURATION_REMAINING),
                getDurationRemainingInSeconds(keyStatus, PLAYBACK_DURATION_REMAINING));
    }

    private static long getDurationRemainingInSeconds(Map<String, String> keyStatus, String property) {
        if (keyStatus != null) {
            try {
                String value = keyStatus.get(property);
                if (value != null) {
                    return Long.parseLong(value);
                }
            } catch (NumberFormatException e) {
                // Ignore
            }
        }
        return C.TIME_UNSET;
    }
} 