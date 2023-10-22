/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.views.text;

import android.graphics.Canvas;
import android.graphics.Paint;
import android.text.style.ReplacementSpan;

/**
 * TextInlineViewSpan is a span for inlined views that are inside <Text/>. It computes
 * its size based on the input size. It contains no draw logic, just positioning logic.
 */
public class TextInlineViewSpan extends ReplacementSpan {
  private final int reactTag;
  private final int width;
  private final int height;

  public TextInlineViewSpan(int reactTag, int width, int height) {
    this.reactTag = reactTag;
    this.width = width;
    this.height = height;
  }

  public int getReactTag() {
    return reactTag;
  }

  public int getWidth() {
    return width;
  }

  public int getHeight() {
    return height;
  }

  @Override
  public int getSize(Paint paint, CharSequence text, int start, int end, Paint.FontMetricsInt fm) {
    // NOTE: This getSize code is copied from DynamicDrawableSpan and modified to not use a Drawable

    if (fm != null) {
      fm.ascent = -height;
      fm.descent = 0;

      fm.top = fm.ascent;
      fm.bottom = 0;
    }

    return width;
  }
}