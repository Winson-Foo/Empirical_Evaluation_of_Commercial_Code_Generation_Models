/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.perftest;

/** PerfTestConfig stub. */
public class PerfTestConfig {

  private static final boolean RUNNING_IN_PERF_TEST = false;

  public static boolean isRunningInPerfTest() {
    return RUNNING_IN_PERF_TEST;
  }
}