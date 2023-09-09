package com.liskovsoft.smartyoutubetv2.tv.ui.playback.actions;

import android.content.Context;
import com.liskovsoft.smartyoutubetv2.tv.R;

/**
 * An action for toggling between 0 and 90 degrees rotation.
 */
public class RotateAction extends TwoStateAction {
    private static final String LABEL_ROTATE_OFF = "0� Rotation";
    private static final String LABEL_ROTATE_ON = "90� Rotation";

    public RotateAction(Context context) {
        super(context, R.id.action_rotate, R.drawable.action_rotate);

        String[] labels = new String[2];
        labels[INDEX_OFF] = LABEL_ROTATE_OFF;
        labels[INDEX_ON] = LABEL_ROTATE_ON;
        setLabels(labels);
    }
}

