/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.views.scroll;

public interface FpsListener {
  /**
   * Start recording data for the specified tag.
   *
   * @param tag The tag to use for recording data.
   */
  void startRecording(String tag);

  /**
   * Stop recording data for the specified tag and report the collected data.
   *
   * <p>Calling this method on a listener that has already been disabled is a no-op.
   *
   * @param tag The tag to use for recording data.
   */
  void stopRecording(String tag);

  /** Returns true if this listener is currently recording data. */
  boolean isRecording();
}

