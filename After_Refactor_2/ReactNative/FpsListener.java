/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.views.scroll;

/**
 * A listener interface for measuring frames per second (FPS).
 */
public interface FpsMeasurementListener {

  /**
   * Enables this listener to begin recording data for the given tag.
   *
   * @param tag the tag to record data for
   */
  void startMeasurementForTag(String tag);

  /**
   * Disables this listener and reports the collected data for the given tag.
   *
   * <p>Calling this method on a listener that has already been disabled is a no-op.
   *
   * @param tag the tag to report data for
   */
  void stopMeasurementForTag(String tag);

  /**
   * Returns whether this listener is currently enabled and recording data.
   *
   * @return true if this listener is enabled, false otherwise
   */
  boolean isMeasurementEnabled();
}

