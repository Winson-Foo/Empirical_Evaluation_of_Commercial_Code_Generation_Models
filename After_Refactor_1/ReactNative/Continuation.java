/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.bridgeless.internal.bolts;

/**
 * A function to be executed after a task has completed.
 */
public interface Continuation<TaskResult, ContinuationResult> {
  
  /**
   * Defines what should happen after a task has completed.
   *
   * @param task The completed task.
   * @return The result of the continuation.
   * @throws Exception If an error occurs while executing the continuation.
   */
  ContinuationResult then(Task<TaskResult> task) throws Exception;
  
}

