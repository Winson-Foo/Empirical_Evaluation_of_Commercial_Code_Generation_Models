/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.views.text;

import android.graphics.Color;
import android.text.TextPaint;
import android.text.style.CharacterStyle;

public class ShadowStyleSpan extends CharacterStyle implements ReactSpan {
  private final float shadowDx, shadowDy, shadowRadius;
  private final int shadowColor;

  // Constructor
  public ShadowStyleSpan(float dx, float dy, float radius, int color) {
    shadowDx = dx;
    shadowDy = dy;
    shadowRadius = radius;
    shadowColor = color;
  }

  // Apply shadow styling to text
  private void applyShadow(TextPaint textPaint) {
    textPaint.setShadowLayer(shadowRadius, shadowDx, shadowDy, shadowColor);
  }

  // Override method in parent class to apply shadow styling
  @Override
  public void updateDrawState(TextPaint textPaint) {
    applyShadow(textPaint);
  }

  // Constants for shadow color and radius
  public static final int SHADOW_COLOR = Color.BLACK;
  public static final float SHADOW_RADIUS = 2f;
}

