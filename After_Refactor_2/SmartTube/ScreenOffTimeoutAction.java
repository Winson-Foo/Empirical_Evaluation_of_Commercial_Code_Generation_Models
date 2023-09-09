package com.liskovsoft.smartyoutubetv2.tv.ui.playback.actions;

import android.content.Context;
import com.liskovsoft.smartyoutubetv2.tv.R;

/**
 * An action for enabling/disabling screen off timeout.
 */
public class ScreenOffTimeoutAction extends TwoStateAction {
    private static final int SCREEN_OFF_TIMEOUT_ACTION_ID = R.id.action_screen_off_timeout;
    private static final int SCREEN_TIMEOUT_ICON_ID = R.drawable.action_screen_timeout_on;
    private static final String SCREEN_OFF_TIMEOUT_LABEL = "player_screen_off_timeout";

    public ScreenOffTimeoutAction(Context context) {
        super(context, SCREEN_OFF_TIMEOUT_ACTION_ID, SCREEN_TIMEOUT_ICON_ID);
        setLabels(getScreenOffTimeoutLabels(context));
    }

    private String[] getScreenOffTimeoutLabels(Context context) {
        String label = context.getString(R.string.SCREEN_OFF_TIMEOUT_LABEL);
        return new String[]{label, label};
    }
} 