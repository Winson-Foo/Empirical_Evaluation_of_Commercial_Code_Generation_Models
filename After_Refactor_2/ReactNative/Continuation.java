/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.bridgeless.internal.bolts;

/**
 * A functional interface that represents a function to be called after a task completes.
 * If the Continuation does not return a Task, but you want the Task to be cancellable, 
 * then throw a CancellationException from the Continuation.
 * 
 * @param <T> The type of the result produced by the Task
 * @param <U> The type of the result produced by the Continuation
 * 
 * @see Task
 */
public interface Continuation<T, U>   { 

  /**
   * Applies the continuation logic to the completed task.
   * 
   * @param task The completed task
   * 
   * @return The result produced by applying the continuation logic to the completed task
   * 
   * @throws Exception Any exception thrown while processing the task
   */
  U apply(Task<T> task) throws Exception;

} 