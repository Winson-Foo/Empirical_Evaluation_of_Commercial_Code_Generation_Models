package com.liskovsoft.smartyoutubetv2.tv.ui.playback.actions;

import android.content.Context;

public class ScreenOffTimeoutAction extends TwoStateAction {
    private static final int TIMEOUT_OPTIONS = 2;
    private static final int TIMEOUT_LABEL = R.string.player_screen_off_timeout;
    private static final int TIMEOUT_ICON_ON = R.drawable.action_screen_timeout_on;

    public ScreenOffTimeoutAction(ScreenOffTimeoutActionBuilder builder) {
        super(builder.context, builder.actionId, builder.iconOn);

        String label = builder.context.getString(TIMEOUT_LABEL);
        String[] labels = new String[TIMEOUT_OPTIONS];

        labels[INDEX_OFF] = label;
        labels[INDEX_ON] = label;
        setLabels(labels);
    }

    public static class ScreenOffTimeoutActionBuilder {
        private final Context context;
        private final int actionId;
        private int iconOn;

        public ScreenOffTimeoutActionBuilder(Context context, int actionId) {
            this.context = context;
            this.actionId = actionId;
        }

        public ScreenOffTimeoutActionBuilder setIconOn(int iconOn) {
            this.iconOn = iconOn;
            return this;
        }

        public ScreenOffTimeoutAction build() {
            return new ScreenOffTimeoutAction(this);
        }
    }
} 