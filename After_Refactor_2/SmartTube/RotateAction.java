package com.liskovsoft.smartyoutubetv2.tv.ui.playback.actions;

import android.content.Context;
import androidx.annotation.DrawableRes;
import androidx.annotation.StringRes;
import com.liskovsoft.smartyoutubetv2.tv.R;

/**
 * An action for toggle between 0 and 90 degrees rotation.
 */
public class RotateAction extends TwoStateAction {

    private static final int OFF_LABEL_RES_ID = R.string.video_rotate;
    private static final int ON_LABEL_RES_ID = R.string.video_rotate;
    private static final int ACTION_ID = R.id.action_rotate;
    private static final int ICON_RES_ID = R.drawable.action_rotate;

    public RotateAction(Context context) {
        super(context, ACTION_ID, ICON_RES_ID);
        setLabels(getLabels(context));
    }

    private String[] getLabels(Context context) {
        String[] labels = new String[2];
        labels[INDEX_OFF] = context.getString(OFF_LABEL_RES_ID);
        labels[INDEX_ON] = context.getString(ON_LABEL_RES_ID);
        return labels;
    }
} 
