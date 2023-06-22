/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.bridgeless.internal.bolts;

import android.annotation.SuppressLint;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/** Provides access to custom thread pools and executors. */
public final class AndroidExecutors {

  /** Singleton instance of this class. */
  private static final AndroidExecutors INSTANCE = new AndroidExecutors();

  /** Thread pool configuration values. */
  private static final int CPU_COUNT = Runtime.getRuntime().availableProcessors();
  private static final int CORE_POOL_SIZE = CPU_COUNT + 1;
  private static final int MAX_POOL_SIZE = CPU_COUNT * 2 + 1;
  private static final long KEEP_ALIVE_TIME = 1L;

  /** Executor for running tasks on the UI thread. */
  private final Executor uiThread;

  /** Creates a new AndroidExecutors instance with the default executor for the UI thread. */
  private AndroidExecutors() {
    uiThread = new UIThreadExecutor();
  }

  /** Returns a cached thread pool with core thread timeout enabled. */
  public static ExecutorService newCachedThreadPool() {
    ThreadPoolExecutor executor =
        new ThreadPoolExecutor(
            CORE_POOL_SIZE,
            MAX_POOL_SIZE,
            KEEP_ALIVE_TIME,
            TimeUnit.SECONDS,
            new LinkedBlockingQueue<Runnable>());

    allowCoreThreadTimeout(executor, true);

    return executor;
  }

  /** Returns a cached thread pool with core thread timeout enabled, using the given thread factory. */
  public static ExecutorService newCachedThreadPool(ThreadFactory threadFactory) {
    ThreadPoolExecutor executor =
        new ThreadPoolExecutor(
            CORE_POOL_SIZE,
            MAX_POOL_SIZE,
            KEEP_ALIVE_TIME,
            TimeUnit.SECONDS,
            new LinkedBlockingQueue<Runnable>(),
            threadFactory);

    allowCoreThreadTimeout(executor, true);

    return executor;
  }

  /** Enables or disables core thread timeout on the given thread pool executor, if supported. */
  @SuppressLint("NewApi")
  public static void allowCoreThreadTimeout(ThreadPoolExecutor executor, boolean value) {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.GINGERBREAD) {
      executor.allowCoreThreadTimeOut(value);
    }
  }

  /**
   * Returns an executor that runs tasks on the UI thread. Runs tasks using a {@link Handler} on the
   * UI thread.
   */
  public static Executor uiThread() {
    return INSTANCE.uiThread;
  }

  /** Executor that runs tasks on the UI thread using a {@link Handler}. */
  private static class UIThreadExecutor implements Executor {
    @Override
    public void execute(Runnable command) {
      new Handler(Looper.getMainLooper()).post(command);
    }
  }
}