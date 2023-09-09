/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.views.unimplementedview;

import android.content.Context;
import android.graphics.Color;
import android.util.AttributeSet;
import android.widget.LinearLayout;
import androidx.annotation.Nullable;
import androidx.appcompat.widget.AppCompatTextView;

public class ReactUnimplementedView extends LinearLayout {
  private AppCompatTextView textView;

  public ReactUnimplementedView(Context context) {
    this(context, null);
  }

  public ReactUnimplementedView(Context context, @Nullable AttributeSet attrs) {
    super(context, attrs);
    initView(context);
  }

  private void initView(Context context) {
    setOrientation(VERTICAL);
    setGravity(Gravity.CENTER_HORIZONTAL);
    setBackgroundColor(0x55ff0000);

    textView = new AppCompatTextView(context);
    textView.setLayoutParams(
        new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT));
    textView.setGravity(Gravity.CENTER);
    textView.setTextColor(Color.WHITE);
    addView(textView);
  }

  public void setName(String name) {
    textView.setText("'" + name + "' is not Fabric compatible yet.");
  }
} 