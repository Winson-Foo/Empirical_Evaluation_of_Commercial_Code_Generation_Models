package com.liskovsoft.smartyoutubetv2.tv.ui.dialogs.other;

import android.content.Context;
import androidx.preference.DialogPreference;
import com.liskovsoft.smartyoutubetv2.common.app.models.playback.ui.ChatReceiver;

public class ChatPreference extends DialogPreference {
    private final ChatReceiver mChatReceiver;

    public ChatPreference(Context context, AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public ChatPreference(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        mChatReceiver = null;
    }

    public ChatPreference(Context context, AttributeSet attrs, int defStyleAttr, int defStyleRes) {
        super(context, attrs, defStyleAttr, defStyleRes);
        mChatReceiver = null;
    }

    public ChatPreference(Context context, ChatReceiver chatReceiver) {
        super(context);
        mChatReceiver = chatReceiver;
    }

    public ChatReceiver getChatReceiver() {
        return mChatReceiver;
    }
}
