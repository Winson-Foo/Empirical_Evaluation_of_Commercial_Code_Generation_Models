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

  private static final WeakHashMap<ReactContext, HeadlessJsTaskContext> INSTANCES =
    new WeakHashMap<>();

  private final WeakReference<ReactContext> mReactContext;
  private final Set<HeadlessJsTaskEventListener> mHeadlessJsTaskEventListeners = new CopyOnWriteArraySet<>();
  private final AtomicInteger mLastTaskId = new AtomicInteger(0);
  private final Handler mHandler = new Handler();
  private final Set<Integer> mActiveTasks = new CopyOnWriteArraySet<>();
  private final Map<Integer, HeadlessJsTaskConfig> mActiveTaskConfigs = new ConcurrentHashMap<>();
  private final SparseArray<Runnable> mTaskTimeouts = new SparseArray<>();

  public static HeadlessJsTaskContext getInstance(ReactContext context) {
    HeadlessJsTaskContext helper = INSTANCES.get(context);
    if (helper == null) {
      helper = new HeadlessJsTaskContext(context);
      INSTANCES.put(context, helper);
    }
    return helper;
  }

  private HeadlessJsTaskContext(ReactContext reactContext) {
    mReactContext = new WeakReference<ReactContext>(reactContext);
  }

  public void addTaskEventListener(HeadlessJsTaskEventListener listener) {
    synchronized (this) {
      mHeadlessJsTaskEventListeners.add(listener);
      for (Integer activeTaskId : mActiveTasks) {
        listener.onHeadlessJsTaskStart(activeTaskId);
      }
    }
  }

  public boolean hasActiveTasks() {
    return !mActiveTasks.isEmpty();
  }

  public int onTaskStart(final HeadlessJsTaskConfig taskConfig) {
    final int taskId = mLastTaskId.incrementAndGet();
    onTaskStart(taskConfig, taskId);
    return taskId;
  }

  private void onTaskStart(final HeadlessJsTaskConfig taskConfig, int taskId) {
    UiThreadUtil.assertOnUiThread();
    ReactContext reactContext =
      Assertions.assertNotNull(
        mReactContext.get(),
        "Tried to start a task on a react context that has already been destroyed"
      );

    if (reactContext.getLifecycleState() == LifecycleState.RESUMED
        && !taskConfig.isAllowedInForeground()) {
      throw new IllegalStateException("Tried to start task " + taskConfig.getTaskKey() + " while in foreground, but this is not allowed.");
    }

    mActiveTasks.add(taskId);
    mActiveTaskConfigs.put(taskId, new HeadlessJsTaskConfig(taskConfig));

    if (reactContext.hasActiveReactInstance()) {
      reactContext.getJSModule(AppRegistry.class).startHeadlessTask(taskId, taskConfig.getTaskKey(), taskConfig.getData());
    } else {
      ReactSoftExceptionLogger.logSoftException(
        "HeadlessJsTaskContext",
        new RuntimeException("Cannot start headless task, CatalystInstance not available")
      );
    }

    if (taskConfig.getTimeout() > 0) {
      scheduleTaskTimeout(taskId, taskConfig.getTimeout());
    }

    for (HeadlessJsTaskEventListener listener : mHeadlessJsTaskEventListeners) {
      listener.onHeadlessJsTaskStart(taskId);
    }
  }

  public boolean onTaskRetry(final int taskId) {
    final HeadlessJsTaskConfig sourceTaskConfig = mActiveTaskConfigs.get(taskId);

    Assertions.assertCondition(
      sourceTaskConfig != null,
      "Tried to retrieve non-existent task config with id " + taskId + "."
    );

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
        retryPolicy.update()
      );

    final Runnable retryAttempt = new Runnable() {
      @Override
      public void run() {
        onTaskStart(taskConfig, taskId);
      }
    };

    UiThreadUtil.runOnUiThread(retryAttempt, retryPolicy.getDelay());
    return true;
  }

  public void onTaskFinish(final int taskId) {
    Assertions.assertCondition(
      mActiveTasks.remove(taskId),
      "Tried to finish non-existent task with id " + taskId + "."
    );

    Assertions.assertCondition(
      mActiveTaskConfigs.remove(taskId) != null,
      "Tried to remove non-existent task config with id " + taskId + "."
    );

    removeTimeout(taskId);

    UiThreadUtil.runOnUiThread(new Runnable() {
      @Override
      public void run() {
        for (HeadlessJsTaskEventListener listener : mHeadlessJsTaskEventListeners) {
          listener.onHeadlessJsTaskFinish(taskId);
        }
      }
    });
  }

  public boolean isTaskRunning(final int taskId) {
    return mActiveTasks.contains(taskId);
  }

  public void removeTaskEventListener(HeadlessJsTaskEventListener listener) {
    mHeadlessJsTaskEventListeners.remove(listener);
  }

  private void removeTimeout(int taskId) {
    Runnable timeout = mTaskTimeouts.get(taskId);
    if (timeout != null) {
      mHandler.removeCallbacks(timeout);
      mTaskTimeouts.remove(taskId);
    }
  }

  private void scheduleTaskTimeout(final int taskId, long timeout) {
    Runnable runnable = new Runnable() {
      @Override
      public void run() {
        onTaskFinish(taskId);
      }
    };
    mTaskTimeouts.append(taskId, runnable);
    mHandler.postDelayed(runnable, timeout);
  }
}