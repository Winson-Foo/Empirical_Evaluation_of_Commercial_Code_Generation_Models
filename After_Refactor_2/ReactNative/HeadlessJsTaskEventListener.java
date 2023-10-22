/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.jstasks;

/**
 * An event listener interface for managing the lifecycle of headless JS tasks.
 */
public interface HeadlessJsTaskEventListener {

  /**
   * Called when a headless JS task is started.
   * 
   * @param id The identifier of the task that was started
   */
  void onHeadlessJsTaskStart(final int id);

  /**
   * Called when a headless JS task is finished.
   * 
   * @param id The identifier of the task that was finished
   */
  void onHeadlessJsTaskFinish(final int id);
} 