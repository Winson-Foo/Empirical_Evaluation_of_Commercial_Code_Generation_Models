package com.facebook.react.views.unimplementedview;

import android.content.Context;
import android.graphics.Color;
import android.view.Gravity;
import android.widget.LinearLayout;

import androidx.appcompat.widget.AppCompatTextView;

import static android.view.ViewGroup.LayoutParams.MATCH_PARENT;
import static android.view.ViewGroup.LayoutParams.WRAP_CONTENT;

public class ReactUnimplementedView extends LinearLayout {
  private final AppCompatTextView textView;

  public ReactUnimplementedView(Context context) {
    super(context);
    textView = createTextView(context);

    setBackgroundColor(0x55ff0000);
    setGravity(Gravity.CENTER_HORIZONTAL);
    setOrientation(LinearLayout.VERTICAL);
    addView(textView);
  }

  public void setName(String name) {
    textView.setText(String.format("'%s' is not Fabric compatible yet.", name));
  }

  private AppCompatTextView createTextView(Context context) {
    AppCompatTextView textView = new AppCompatTextView(context);
    textView.setLayoutParams(new LinearLayout.LayoutParams(WRAP_CONTENT, MATCH_PARENT));
    textView.setGravity(Gravity.CENTER);
    textView.setTextColor(Color.WHITE);
    return textView;
  }
} 

