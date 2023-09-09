package com.liskovsoft.smartyoutubetv2.tv.ui.playback.actions;

import android.content.Context;
import com.liskovsoft.smartyoutubetv2.tv.R;

/**
 * An action for toggling screen off timeout.
 */
public class ScreenOffTimeoutToggleAction extends TwoStateAction {
    private static final int STATE_OFF = 0;
    private static final int STATE_ON = 1;

    private static final String LABEL = "player_screen_off_timeout";

    public ScreenOffTimeoutToggleAction(Context context) {
        super(context, R.id.action_screen_off_timeout, R.drawable.action_screen_timeout_on);

        String label = context.getString(R.string.LABEL);
        String[] labels = new String[2];
        labels[STATE_OFF] = label;
        labels[STATE_ON] = label;
        setLabels(labels);
    }
}

