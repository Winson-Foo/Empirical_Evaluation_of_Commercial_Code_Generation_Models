/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.jstasks;

/**
 * Listener interface for events related to headless JS tasks.
 */
public interface HeadlessJsTaskEventListener {

  /**
   * Invoked when a headless JS task is started.
   *
   * @param taskId The unique identifier of the started task
   */
  void onHeadlessJsTaskStarted(int taskId);

  /**
   * Invoked when a headless JS task finishes.
   *
   * @param taskId The unique identifier of the finished task
   */
  void onHeadlessJsTaskFinished(int taskId);
}