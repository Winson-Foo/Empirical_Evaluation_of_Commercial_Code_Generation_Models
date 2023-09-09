package com.facebook.react.uimanager;

import android.view.View;

public class IllegalViewOperationException extends RuntimeException {

    private final View view;

    public IllegalViewOperationException(String message, View view) {
        super(message);
        this.view = view;
    }

    public IllegalViewOperationException(String message) {
        this(message, null);
    }

    public View getView() {
        return view;
    }
}
