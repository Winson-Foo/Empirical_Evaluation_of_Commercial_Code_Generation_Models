package com.liskovsoft.smartyoutubetv2.tv.ui.playback.actions;

import android.content.Context;
import com.liskovsoft.smartyoutubetv2.tv.R;

public class PlayerTweaksAction extends TwoStateAction {

    private static final int ACTION_ID = R.id.action_video_stats;
    private static final int ICON_ID = R.drawable.action_video_stats;
    private static final String OFF_LABEL = "Player Tweaks";
    private static final String ON_LABEL = "Player Tweaks";

    public PlayerTweaksAction(Context context) {
        super(context, ACTION_ID, ICON_ID);
        setLabels(getLabels(context));
    }

    private String[] getLabels(Context context) {
        String[] labels = new String[2];
        labels[INDEX_OFF] = context.getString(R.string.player_tweaks);
        labels[INDEX_ON] = context.getString(R.string.player_tweaks);
        return labels;
    }
} 