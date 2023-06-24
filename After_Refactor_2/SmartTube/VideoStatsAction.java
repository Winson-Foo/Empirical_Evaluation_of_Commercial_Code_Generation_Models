package com.liskovsoft.smartyoutubetv2.tv.ui.playback.actions;

import android.content.Context;
import com.liskovsoft.smartyoutubetv2.tv.R;

public class VideoStatsAction extends TwoStateAction {
    private static final int ACTION_ID = R.id.action_video_stats;
    private static final int ICON_ID = R.drawable.action_video_stats;
    private static final String LABEL_TEXT = "player_tweaks";
    // Note: use the same label for both states, as they represent the same action

    public VideoStatsAction(Context context) {
        super(context, ACTION_ID, ICON_ID);

        String[] labels = new String[2];
        labels[INDEX_OFF] = context.getString(R.string[LABEL_TEXT]);
        labels[INDEX_ON] = context.getString(R.string[LABEL_TEXT]);
        setLabels(labels);
    }
}

