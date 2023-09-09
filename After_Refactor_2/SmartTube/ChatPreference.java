package com.liskovsoft.smartyoutubetv2.tv.ui.dialogs.other;

import android.content.Context;
import android.util.AttributeSet;

import androidx.preference.DialogPreference;

import com.liskovsoft.smartyoutubetv2.common.app.models.playback.ui.ChatReceiver;

public class ChatPreference extends DialogPreference {

    private ChatReceiver mChatReceiver;

    public ChatPreference(Context context, AttributeSet attrs, int defStyleAttr, int defStyleRes) {
        super(context, attrs, defStyleAttr, defStyleRes);
        init();
    }

    public ChatPreference(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    public ChatPreference(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public ChatPreference(Context context) {
        super(context);
        init();
    }

    private void init() {
        setDialogLayoutResource(R.layout.chat_preference_dialog_layout);
        setPositiveButtonText(android.R.string.ok);
        setNegativeButtonText(android.R.string.cancel);
        setDialogIcon(null);
    }

    public void setChatReceiver(ChatReceiver chatReceiver) {
        mChatReceiver = chatReceiver;
    }

    public ChatReceiver getChatReceiver() {
        return mChatReceiver;
    }
} 