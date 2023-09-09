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
 * TextInlineViewPlaceholderSpan is a ReplacementSpan for inline views inside <Text/> that computes its size based on input size.
 * It contains no draw logic, just positioning logic.
 * Use this span to insert inline views in Text component.
 */
public class TextInlineViewPlaceholderSpan extends ReplacementSpan implements ReactSpan {

  private final int mReactTag;
  private final int mWidth;
  private final int mHeight;

  public TextInlineViewPlaceholderSpan(int reactTag, int width, int height) {
    mReactTag = reactTag;
    mWidth = width;
    mHeight = height;
  }

  /** Returns the ID of the React view this span represents */
  public int getReactTag() {
    return mReactTag;
  }

  /** Returns the width of the placeholder for the inline view */
  public int getWidth() {
    return mWidth;
  }

  /** Returns the height of the placeholder for the inline view*/
  public int getHeight() {
    return mHeight;
  }

  /**
   * Computes the dimensions of the span according to the width and height passed in the constructor.
   * It updates the FontMetricsInt to enable proper vertical alignment with the text baseline.
   * @return the width of the span
   */
  @Override
  public int getSize(Paint paint, CharSequence text, int start, int end, Paint.FontMetricsInt fm) {
    if (fm != null) {
      fm.ascent = -mHeight;
      fm.descent = 0;
      fm.top = fm.ascent;
      fm.bottom = 0;
    }

    return mWidth;
  }

  /** Draws nothing */
  @Override
  public void draw(Canvas canvas, CharSequence text, int start, int end, float x, int top, int y, int bottom, Paint paint) {}
  
} 