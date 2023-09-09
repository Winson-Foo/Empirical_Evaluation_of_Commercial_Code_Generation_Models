// ContinuationFunction.java

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.bridgeless.internal.bolts;

/**
 * A function to be called after a task completes.
 *
 * <p>If you wish to have the Task from a Continuation that does not return a Task be cancelled then
 * throw a {@link java.util.concurrent.CancellationException} from the Continuation.
 *
 * @see Task
 */
public interface ContinuationFunction<TTaskResult, TContinuationResult> {
  
  /**
   * This method is called when a task completes.
   *
   * @param task The completed task.
   * @return The result of the continuation.
   * @throws Exception If there was an error during the continuation.
   */
  TContinuationResult apply(Task<TTaskResult> task) throws Exception;
}

// TaskContinuationFunction.java

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.bridgeless.internal.bolts;

/**
 * A function to be called after a task completes.
 *
 * <p>If you wish to have the Task from a Continuation that does not return a Task be cancelled then
 * throw a {@link java.util.concurrent.CancellationException} from the Continuation.
 *
 * @see Task
 */
public interface TaskContinuationFunction<TTaskResult, TContinuationResult> {
  
  /**
   * This method is called when a task completes.
   *
   * @param task The completed task.
   * @return The result of the continuation.
   * @throws Exception If there was an error during the continuation.
   */
  TContinuationResult apply(Task<TTaskResult> task) throws Exception;
}

