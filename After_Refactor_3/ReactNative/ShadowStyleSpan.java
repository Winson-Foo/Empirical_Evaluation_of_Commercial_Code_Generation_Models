/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.views.text;

import android.text.TextPaint;
import android.text.style.CharacterStyle;

/**
 * Represents the shadow style span of a text. 
 * This is a character style that adds a shadow to the text.
 */
public class ShadowStyleSpan extends CharacterStyle implements ReactSpan {
  private final float deltaX;
  private final float deltaY;
  private final float radius;
  private final int color;

  /**
   * Constructor to create ShadowStyleSpan.
   * @param deltaX The X-axis offset of the shadow.
   * @param deltaY The Y-axis offset of the shadow.
   * @param radius The radius of the shadow.
   * @param color The color of the shadow.
   */
  public ShadowStyleSpan(final float deltaX, final float deltaY, final float radius, final int color) {
    this.deltaX = deltaX;
    this.deltaY = deltaY;
    this.radius = radius;
    this.color = color;
  }

  /**
   * Applies the shadow style to the given text paint.
   * @param textPaint The text paint to apply shadow to.
   */
  @Override
  public void updateDrawState(final TextPaint textPaint) {
    textPaint.setShadowLayer(radius, deltaX, deltaY, color);
  }

  /**
  * Checks if this is equal to the given object.
  * @param o The object to compare with.
  * @return True if equal, false otherwise.
  */
  @Override
  public boolean equals(final Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    final ShadowStyleSpan that = (ShadowStyleSpan) o;
    if (Float.compare(that.deltaX, deltaX) != 0) return false;
    if (Float.compare(that.deltaY, deltaY) != 0) return false;
    if (Float.compare(that.radius, radius) != 0) return false;
    return color == that.color;
  }

  /**
  * Generates the hashcode for this object.
  * @return The hashcode.
  */
  @Override
  public int hashCode() {
    int result = (deltaX != +0.0f ? Float.floatToIntBits(deltaX) : 0);
    result = 31 * result + (deltaY != +0.0f ? Float.floatToIntBits(deltaY) : 0);
    result = 31 * result + (radius != +0.0f ? Float.floatToIntBits(radius) : 0);
    result = 31 * result + color;
    return result;
  }
}

