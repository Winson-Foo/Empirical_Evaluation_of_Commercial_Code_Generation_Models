package com.google.android.exoplayer2.drm;

import android.util.Pair;

public final class WidevineUtil {

    public static final String PROPERTY_LICENSE_DURATION_REMAINING = "LicenseDurationRemaining";
    public static final String PROPERTY_PLAYBACK_DURATION_REMAINING = "PlaybackDurationRemaining";

    private WidevineUtil() {}

    public static Pair<Long, Long> getLicenseDurationRemainingSec(DrmSessionWrapper drmSession) {
        Map<String, String> keyStatus = drmSession.queryKeyStatus();
        if (keyStatus == null) {
            return null;
        }
        long licenseDurationRemaining = DurationUtil.getDurationRemainingSec(keyStatus, PROPERTY_LICENSE_DURATION_REMAINING);
        long playbackDurationRemaining = DurationUtil.getDurationRemainingSec(keyStatus, PROPERTY_PLAYBACK_DURATION_REMAINING);
        return new Pair<>(licenseDurationRemaining, playbackDurationRemaining);
    }

} 