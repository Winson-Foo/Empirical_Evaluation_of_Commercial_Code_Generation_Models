package com.liskovsoft.smartyoutubetv2.tv.ui.dialogs.other;

import android.content.Context;
import androidx.preference.DialogPreference;
import com.liskovsoft.smartyoutubetv2.common.app.models.playback.ui.CommentsReceiver;

public class CommentsDialogPreference extends DialogPreference {

    private CommentsReceiver mCommentsReceiver;

    public CommentsDialogPreference(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public CommentsDialogPreference(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    public CommentsDialogPreference(Context context, AttributeSet attrs, int defStyleAttr, int defStyleRes) {
        super(context, attrs, defStyleAttr, defStyleRes);
    }

    public void setCommentsReceiver(CommentsReceiver commentsReceiver) {
        mCommentsReceiver = commentsReceiver;
    }

    public void setDialogTitle(CharSequence title) {
        super.setDialogTitle(title);
    }

}

