package com.liskovsoft.smartyoutubetv2.tv.ui.dialogs.other;

import android.content.Context;
import android.util.AttributeSet;
import androidx.preference.DialogPreference;
import com.liskovsoft.smartyoutubetv2.common.app.models.playback.ui.CommentsReceiver;

/**
 * A preference that allows setting a {@link CommentsReceiver} to receive comments.
 */
public class CommentsReceiverPreference extends DialogPreference {
    private final CommentsReceiver mCommentsReceiver;

    /**
     * Constructor for creating a new CommentsReceiverPreference.
     * 
     * @param context The context.
     * @param attrs The attribute set.
     * @param defStyleAttr The default style attribute.
     * @param defStyleRes The default style resource.
     * @param commentsReceiver The comments receiver.
     */
    public CommentsReceiverPreference(Context context, AttributeSet attrs, int defStyleAttr, int defStyleRes, CommentsReceiver commentsReceiver) {
        super(context, attrs, defStyleAttr, defStyleRes);

        mCommentsReceiver = commentsReceiver;
    }

    /**
     * Constructor for creating a new CommentsReceiverPreference with default style attribute and resource.
     * 
     * @param context The context.
     * @param attrs The attribute set.
     * @param commentsReceiver The comments receiver.
     */
    public CommentsReceiverPreference(Context context, AttributeSet attrs, CommentsReceiver commentsReceiver) {
        this(context, attrs, android.R.attr.dialogPreferenceStyle, android.R.style.Theme_Material_Dialog, commentsReceiver);
    }

    /**
     * Gets the comments receiver.
     * 
     * @return The comments receiver.
     */
    public CommentsReceiver getCommentsReceiver() {
        return mCommentsReceiver;
    }
}