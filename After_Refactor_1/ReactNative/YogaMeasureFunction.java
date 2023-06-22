/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. 
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.yoga;

/**
 * Interface for measuring the dimensions of a YogaNode.
 */
public interface YogaMeasureFunction {
  /**
   * Measure the YogaNode and return the dimensions. 
   *
   * @param node the YogaNode to be measured.
   * @param width the width to measure the node.
   * @param widthMode the width mode to measure the node.
   * @param height the height to measure the node.
   * @param heightMode the height mode to measure the node.
   * @return the dimensions of the measured node.
   */
  long measure(
      YogaNode node,
      float width,
      YogaMeasureMode widthMode,
      float height,
      YogaMeasureMode heightMode
  );
}