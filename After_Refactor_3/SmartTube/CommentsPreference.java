package com.liskovsoft.smartyoutubetv2.tv.ui.dialogs.other;

import android.content.Context;
import android.util.AttributeSet;
import androidx.preference.DialogPreference;
import com.liskovsoft.smartyoutubetv2.common.app.models.playback.ui.CommentsReceiver;

/**
 * A preference that allows the user to view and post comments.
 */
public class CommentsPreference extends DialogPreference {

    private CommentsReceiver mCommentsReceiver;

    /**
     * Creates a new CommentsPreference instance.
     * @param context The context.
     * @param attrs The attribute set.
     * @param defStyleAttr The default style attribute.
     * @param defStyleRes The default style resource.
     */
    public CommentsPreference(Context context, AttributeSet attrs, int defStyleAttr, int defStyleRes) {
        super(context, attrs, defStyleAttr, defStyleRes);
    }

    /**
     * Creates a new CommentsPreference instance.
     * @param context The context.
     * @param attrs The attribute set.
     * @param defStyleAttr The default style attribute.
     */
    public CommentsPreference(Context context, AttributeSet attrs, int defStyleAttr) {
        this(context, attrs, defStyleAttr, 0);
    }

    /**
     * Creates a new CommentsPreference instance.
     * @param context The context.
     * @param attrs The attribute set.
     */
    public CommentsPreference(Context context, AttributeSet attrs) {
        this(context, attrs, androidx.preference.R.attr.dialogPreferenceStyle);
    }

    /**
     * Creates a new CommentsPreference instance.
     * @param context The context.
     */
    public CommentsPreference(Context context) {
        this(context, null);
    }

    /**
     * Sets the comments receiver.
     * @param commentsReceiver The comments receiver.
     */
    public void setCommentsReceiver(CommentsReceiver commentsReceiver) {
        mCommentsReceiver = commentsReceiver;
    }

    /**
     * Gets the comments receiver.
     * @return The comments receiver.
     */
    public CommentsReceiver getCommentsReceiver() {
        return mCommentsReceiver;
    }
} 