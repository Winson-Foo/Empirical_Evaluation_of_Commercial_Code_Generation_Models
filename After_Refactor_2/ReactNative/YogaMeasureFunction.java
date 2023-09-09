/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.yoga;

/**
 * Interface to measure the dimensions of a YogaNode.
 */
public interface YogaMeasureFunction {

  /**
   * Measures the dimensions of a YogaNode based on the given parameters.
   *
   * @param node the YogaNode to be measured.
   * @param width the width of the YogaNode to be measured.
   * @param widthMode the mode in which the width is measured (default, exactly, at most).
   * @param height the height of the YogaNode to be measured.
   * @param heightMode the mode in which the height is measured (default, exactly, at most).
   * @return a long value created by YogaMeasureOutput.make(width, height).
   */
  long measure(
      YogaNode node,
      float width,
      YogaMeasureMode widthMode,
      float height,
      YogaMeasureMode heightMode);
}

