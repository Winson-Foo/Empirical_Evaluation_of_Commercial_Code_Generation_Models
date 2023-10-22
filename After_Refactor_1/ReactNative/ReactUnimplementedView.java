package com.facebook.react.views.unimplementedview;

import android.content.Context;
import android.graphics.Color;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.LinearLayout;

import androidx.appcompat.widget.AppCompatTextView;

public class UnimplementedView extends ViewGroup {
  private static final int COLOR_TRANSPARENT_RED = 0x55ff0000;
  private static final int WIDTH_WRAP_CONTENT = LinearLayout.LayoutParams.WRAP_CONTENT;
  private static final int HEIGHT_MATCH_PARENT = LinearLayout.LayoutParams.MATCH_PARENT;

  private final AppCompatTextView unimplementedTextView;

  public UnimplementedView(Context context) {
    super(context);
    setLayoutParams(new LayoutParams(WIDTH_WRAP_CONTENT, HEIGHT_MATCH_PARENT));

    setBackgroundColor(COLOR_TRANSPARENT_RED);
    setGravity(Gravity.CENTER_HORIZONTAL);
    setOrientation(LinearLayout.VERTICAL);

    unimplementedTextView = createUnimplementedTextView(context);
    addView(unimplementedTextView);
  }

  public void setUnimplementedViewName(String name) {
    unimplementedTextView.setText("'" + name + "' is not Fabric compatible yet.");
  }

  private AppCompatTextView createUnimplementedTextView(Context context) {
    AppCompatTextView textView = new AppCompatTextView(context);
    textView.setLayoutParams(new LayoutParams(WIDTH_WRAP_CONTENT, HEIGHT_MATCH_PARENT));
    textView.setGravity(Gravity.CENTER);
    textView.setTextColor(Color.WHITE);
    return textView;
  }

  @Override
  protected void onLayout(boolean changed, int l, int t, int r, int b) {
    // Do nothing
  }
}