/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.perftest;

/** 
 * This class represents the configuration for performance testing.
 */
public class PerformanceTestConfig {

  /**
   * Checks if the application is currently running in a performance test environment.
   * 
   * @return true if running in a performance test, false otherwise
   */
  public boolean isRunningInPerformanceTest() {
    return false;
  }
}

