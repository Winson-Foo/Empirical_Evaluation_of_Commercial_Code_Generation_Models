package com.liskovsoft.smartyoutubetv2.tv.ui.playback.actions;

import android.content.Context;
import androidx.annotation.DrawableRes;
import androidx.annotation.IdRes;

import com.liskovsoft.smartyoutubetv2.tv.R;

/**
 * Action to toggle video stats display
 */
public class ToggleVideoStatsAction extends TwoStateAction {

    private static final String LABEL_OFF = "Player Tweaks";
    private static final String LABEL_ON = "Player Tweaks";

    public ToggleVideoStatsAction(Context context) {
        super(context, R.id.action_toggle_video_stats, R.drawable.action_toggle_video_stats);
        setLabels(getBuilder().build());
    }

    private Builder getBuilder() {
        return new Builder()
                .withOffLabel(LABEL_OFF)
                .withOnLabel(LABEL_ON);
    }

    private static class Builder {

        private String offLabel;
        private String onLabel;

        public Builder withOffLabel(String offLabel) {
            this.offLabel = offLabel;
            return this;
        }

        public Builder withOnLabel(String onLabel) {
            this.onLabel = onLabel;
            return this;
        }

        public String[] build() {
            return new String[]{offLabel, onLabel};
        }
    }
}

