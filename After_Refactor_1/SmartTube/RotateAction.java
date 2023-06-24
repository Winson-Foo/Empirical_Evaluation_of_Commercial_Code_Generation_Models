package com.liskovsoft.smartyoutubetv2.tv.ui.playback.actions;

import android.content.Context;
import com.liskovsoft.smartyoutubetv2.tv.R;

/**
 * An action for toggle between 0 and 90 degrees rotation.
 */
public class RotateAction extends TwoStateAction {
    private static final String LABEL_ROTATE = "Rotate";

    private enum RotateState {
        OFF(R.drawable.action_rotate_off),
        ON(R.drawable.action_rotate_on);

        private final int drawableId;

        RotateState(int drawableId) {
            this.drawableId = drawableId;
        }
    }

    public RotateAction(Context context) {
        super(context, R.id.action_rotate, RotateState.OFF.drawableId, RotateState.ON.drawableId);

        String[] labels = new String[2];
        labels[RotateState.OFF.ordinal()] = LABEL_ROTATE;
        labels[RotateState.ON.ordinal()] = LABEL_ROTATE;
        setLabels(labels);
    }
}

