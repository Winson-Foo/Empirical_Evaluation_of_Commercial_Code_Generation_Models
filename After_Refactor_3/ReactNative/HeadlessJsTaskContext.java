package com.facebook.react.jstasks;

import android.os.Handler;
import android.util.SparseArray;
import com.facebook.infer.annotation.Assertions;
import com.facebook.react.bridge.ReactContext;
import com.facebook.react.bridge.ReactSoftExceptionLogger;
import com.facebook.react.bridge.UiThreadUtil;
import com.facebook.react.common.LifecycleState;
import com.facebook.react.modules.appregistry.AppRegistry;
import java.lang.ref.WeakReference;
import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArraySet;
import java.util.concurrent.atomic.AtomicInteger;

public class HeadlessJsTaskContext {

  private static final WeakHashMap<ReactContext, HeadlessJsTaskContext> INSTANCES = new WeakHashMap<>();
  private static final int DEFAULT_TASK_TIMEOUT = 0;
  private static final int RESUMED = LifecycleState.RESUMED.ordinal();

  public static HeadlessJsTaskContext getInstance(ReactContext context) {
    HeadlessJsTaskContext helper = INSTANCES.get(context);
    if (helper == null) {
      helper = new HeadlessJsTaskContext(context);
      INSTANCES.put(context, helper);
    }
    return helper;
  }

  private final WeakReference<ReactContext> mReactContext;
  private final Set<HeadlessJsTaskEventListener> mHeadlessJsTaskEventListeners = new CopyOnWriteArraySet<>();
  private final AtomicInteger mLastTaskId = new AtomicInteger(0);
  private final Handler mHandler = new Handler();
  private final Set<Integer> mActiveTasks = new CopyOnWriteArraySet<>();
  private final Map<Integer, HeadlessJsTaskConfig> mActiveTaskConfigs = new ConcurrentHashMap<>();
  private final SparseArray<Runnable> mTaskTimeouts = new SparseArray<>();

  private HeadlessJsTaskContext(ReactContext reactContext) {
    mReactContext = new WeakReference<>(reactContext);
  }

  public synchronized void addTaskEventListener(HeadlessJsTaskEventListener listener) {
    mHeadlessJsTaskEventListeners.add(listener);
    for (Integer activeTaskId : mActiveTasks) {
      listener.onHeadlessJsTaskStart(activeTaskId);
    }
  }

  public void removeTaskEventListener(HeadlessJsTaskEventListener listener) {
    mHeadlessJsTaskEventListeners.remove(listener);
  }

  public boolean hasActiveTasks() {
    return !mActiveTasks.isEmpty();
  }

  public synchronized int startTask(final HeadlessJsTaskConfig taskConfig) {
    final int taskId = mLastTaskId.incrementAndGet();
    startTask(taskConfig, taskId);
    return taskId;
  }

  private synchronized void startTask(final HeadlessJsTaskConfig taskConfig, int taskId) {
    UiThreadUtil.assertOnUiThread();
    ReactContext reactContext = Assertions.assertNotNull(
        mReactContext.get(),
        "Tried to start a task on a react context that has already been destroyed");
    if (reactContext.getLifecycleState() == RESUMED && !taskConfig.isAllowedInForeground()) {
      throw new IllegalStateException(
          "Tried to start task " + taskConfig.getTaskKey() + " while in foreground, but this is not allowed.");
    }
    mActiveTasks.add(taskId);
    mActiveTaskConfigs.put(taskId, new HeadlessJsTaskConfig(taskConfig));
    if (reactContext.hasActiveReactInstance()) {
      reactContext.getJSModule(AppRegistry.class).startHeadlessTask(taskId, taskConfig.getTaskKey(), taskConfig.getData());
    } else {
      ReactSoftExceptionLogger.logSoftException(
          "HeadlessJsTaskContext",
          new RuntimeException("Cannot start headless task, CatalystInstance not available"));
    }
    if (taskConfig.getTimeout() > DEFAULT_TASK_TIMEOUT) {
      scheduleTaskTimeout(taskId, taskConfig.getTimeout());
    }
    mHeadlessJsTaskEventListeners.forEach(listener -> listener.onHeadlessJsTaskStart(taskId));
  }

  public synchronized boolean retryTask(final int taskId) {
    final HeadlessJsTaskConfig sourceTaskConfig = mActiveTaskConfigs.get(taskId);
    Assertions.assertCondition(
        sourceTaskConfig != null,
        "Tried to retrieve non-existent task config with id " + taskId + ".");

    final HeadlessJsTaskRetryPolicy retryPolicy = sourceTaskConfig.getRetryPolicy();
    if (!retryPolicy.canRetry()) {
      return false;
    }

    removeTimeout(taskId);
    final HeadlessJsTaskConfig taskConfig =
        new HeadlessJsTaskConfig(
            sourceTaskConfig.getTaskKey(),
            sourceTaskConfig.getData(),
            sourceTaskConfig.getTimeout(),
            sourceTaskConfig.isAllowedInForeground(),
            retryPolicy.update());

    final Runnable retryAttempt = () -> startTask(taskConfig, taskId);
    UiThreadUtil.runOnUiThread(retryAttempt, retryPolicy.getDelay());
    return true;
  }

  public synchronized void finishTask(final int taskId) {
    Assertions.assertCondition(
        mActiveTasks.remove(taskId),
        "Tried to finish non-existent task with id " + taskId + ".");
    Assertions.assertCondition(
        mActiveTaskConfigs.remove(taskId) != null,
        "Tried to remove non-existent task config with id " + taskId + ".");
    removeTimeout(taskId);
    UiThreadUtil.runOnUiThread(() -> mHeadlessJsTaskEventListeners.forEach(listener -> listener.onHeadlessJsTaskFinish(taskId)));
  }

  public synchronized boolean isTaskRunning(final int taskId) {
    return mActiveTasks.contains(taskId);
  }

  private void removeTimeout(int taskId) {
    Runnable timeout = mTaskTimeouts.get(taskId);
    if (timeout != null) {
      mHandler.removeCallbacks(timeout);
      mTaskTimeouts.remove(taskId);
    }
  }

  private void scheduleTaskTimeout(final int taskId, long timeout) {
    Runnable runnable = () -> finishTask(taskId);
    mTaskTimeouts.append(taskId, runnable);
    mHandler.postDelayed(runnable, timeout);
  }
}