/* 
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * 
 * This source code is licensed under the MIT license found in the 
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.yoga;

/**
 * Interface for measuring Yoga layout nodes.
 */
public interface YogaMeasureFunction {

  /**
   * Measures the size of a Yoga layout node.
   * 
   * @param node The node to measure.
   * @param width The width of the node.
   * @param widthMode The width mode of the node.
   * @param height The height of the node.
   * @param heightMode The height mode of the node.
   * @return A value created by YogaMeasureOutput.make(width, height).
   */
  long measure(YogaNode node, 
               float width, 
               YogaMeasureMode widthMode, 
               float height, 
               YogaMeasureMode heightMode);
}

