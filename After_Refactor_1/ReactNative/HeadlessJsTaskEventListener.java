/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.jstasks;

/** Listener interface for task lifecycle events. */
public interface TaskEventListener {

  /**
   * Called when a Task is started.
   *
   * @param taskId A unique identifier for the task instance
   */
  void onTaskStart(int taskId);

  /**
   * Called when a Task finishes.
   *
   * @param taskId A unique identifier for the task instance
   */
  void onTaskFinish(int taskId);
}