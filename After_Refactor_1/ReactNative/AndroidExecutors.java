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

public final class AndroidExecutors {
    private static final AndroidExecutors INSTANCE = new AndroidExecutors();

    private final Executor uiThread;

    private AndroidExecutors() {
        uiThread = new UIThreadExecutor();
    }

    private static ThreadPoolExecutor createThreadPoolExecutor(ThreadFactory threadFactory) {
        ThreadPoolExecutor executor = new ThreadPoolExecutor(
                AndroidThreadConfig.CORE_POOL_SIZE,
                AndroidThreadConfig.MAX_POOL_SIZE,
                AndroidThreadConfig.KEEP_ALIVE_TIME,
                TimeUnit.SECONDS,
                new LinkedBlockingQueue<Runnable>(),
                threadFactory);

        allowCoreThreadTimeout(executor, true);

        return executor;
    }

    public static ExecutorService newCachedThreadPool() {
        return createThreadPoolExecutor(Executors.defaultThreadFactory());
    }

    public static ExecutorService newCachedThreadPool(ThreadFactory threadFactory) {
        return createThreadPoolExecutor(threadFactory);
    }

    @SuppressLint("NewApi")
    public static void allowCoreThreadTimeout(ThreadPoolExecutor executor, boolean value) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.GINGERBREAD) {
            executor.allowCoreThreadTimeOut(value);
        }
    }

    public static Executor uiThread() {
        return INSTANCE.uiThread;
    }

    private static class UIThreadExecutor implements Executor {
        @Override
        public void execute(Runnable command) {
            new Handler(Looper.getMainLooper()).post(command);
        }
    }
}

final class AndroidThreadConfig {
    private AndroidThreadConfig() {}

    static final int CPU_COUNT = Runtime.getRuntime().availableProcessors();
    static final int CORE_POOL_SIZE = CPU_COUNT + 1;
    static final int MAX_POOL_SIZE = CPU_COUNT * 2 + 1;
    static final long KEEP_ALIVE_TIME = 1L;
}

final class Executors {
    private Executors() {}

    static ThreadFactory defaultThreadFactory() {
        return new DefaultThreadFactory();
    }

    private static class DefaultThreadFactory implements ThreadFactory {
        static final String THREAD_NAME_PREFIX = "android-executor-";

        private static int mCount = 0;

        @Override
        public Thread newThread(Runnable r) {
            return new Thread(r, THREAD_NAME_PREFIX + mCount++);
        }
    }
}