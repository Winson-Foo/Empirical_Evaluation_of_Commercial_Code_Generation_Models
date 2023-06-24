public interface HeadlessJsTaskEventListener {
  void onHeadlessJsTaskStart(int taskId);
  void onHeadlessJsTaskFinish(int taskId);
}

public class HeadlessJsTaskContext {
  private static final Map<ReactContext, HeadlessJsTaskContext> instances = new WeakHashMap<>();

  public static HeadlessJsTaskContext getInstance(ReactContext context) {
    HeadlessJsTaskContext instance = instances.get(context);
    if (instance == null) {
      instance = new HeadlessJsTaskContext(context);
      instances.put(context, instance);
    }
    return instance;
  }

  private final WeakReference<ReactContext> reactContext;
  private final Set<HeadlessJsTaskEventListener> eventListeners = new HashSet<>();
  private final AtomicInteger lastTaskId = new AtomicInteger(0);
  private final Handler handler = new Handler();
  private final Set<Integer> activeTasks = new HashSet<>();
  private final Map<Integer, HeadlessJsTaskConfig> activeTaskConfigs = new ConcurrentHashMap<>();
  private final SparseArray<Runnable> taskTimeouts = new SparseArray<>();

  private HeadlessJsTaskContext(ReactContext reactContext) {
    this.reactContext = new WeakReference<>(reactContext);
  }

  public void addEventListener(HeadlessJsTaskEventListener listener) {
    eventListeners.add(listener);
    activeTasks.forEach(taskId -> listener.onHeadlessJsTaskStart(taskId));
  }

  public void removeEventListener(HeadlessJsTaskEventListener listener) {
    eventListeners.remove(listener);
  }

  public boolean hasActiveTasks() {
    return !activeTasks.isEmpty();
  }

  public int startTask(HeadlessJsTaskConfig config) {
    int taskId = lastTaskId.incrementAndGet();
    startTask(config, taskId);
    return taskId;
  }

  public void finishTask(int taskId) {
    if (!activeTasks.contains(taskId)) {
      throw new IllegalArgumentException("No such task with ID: " + taskId);
    }

    activeTasks.remove(taskId);
    activeTaskConfigs.remove(taskId);
    removeTimeout(taskId);

    for (HeadlessJsTaskEventListener listener : eventListeners) {
      listener.onHeadlessJsTaskFinish(taskId);
    }
  }

  public void retryTask(int taskId) {
    HeadlessJsTaskConfig config = activeTaskConfigs.get(taskId);

    if (config == null) {
      throw new IllegalArgumentException("No such task with ID: " + taskId);
    }

    HeadlessJsTaskRetryPolicy retryPolicy = config.getRetryPolicy();

    if (!retryPolicy.canRetry()) {
      return;
    }

    removeTimeout(taskId);

    int newTaskId = lastTaskId.incrementAndGet();
    HeadlessJsTaskConfig newConfig = new HeadlessJsTaskConfig(
      config.getTaskKey(),
      config.getData(),
      config.getTimeout(),
      config.isAllowedInForeground(),
      retryPolicy.update()
    );

    startTask(newConfig, newTaskId);

    for (HeadlessJsTaskEventListener listener : eventListeners) {
      listener.onHeadlessJsTaskRetry(newTaskId, retryPolicy);
    }
  }

  private void startTask(HeadlessJsTaskConfig config, int taskId) {
    UiThreadUtil.assertOnUiThread();

    if (reactContext.get().getLifecycleState() == LifecycleState.RESUMED && !config.isAllowedInForeground()) {
      throw new IllegalStateException("Cannot start task while in foreground");
    }

    ReactContext reactContext = this.reactContext.get();

    if (reactContext == null) {
      return;
    }

    activeTasks.add(taskId);
    activeTaskConfigs.put(taskId, new HeadlessJsTaskConfig(config));

    if (reactContext.hasActiveReactInstance()) {
      reactContext.getJSModule(AppRegistry.class).startHeadlessTask(taskId, config.getTaskKey(), config.getData());
    } else {
      ReactSoftExceptionLogger.logSoftException("HeadlessJsTaskContext", new RuntimeException("Cannot start headless task, CatalystInstance not available"));
    }

    if (config.getTimeout() > 0) {
      scheduleTaskTimeout(taskId, config.getTimeout());
    }

    for (HeadlessJsTaskEventListener listener : eventListeners) {
      listener.onHeadlessJsTaskStart(taskId);
    }
  }

  public boolean isTaskRunning(int taskId) {
    return activeTasks.contains(taskId);
  }

  private void scheduleTaskTimeout(int taskId, long timeout) {
    Runnable runnable = () -> finishTask(taskId);

    taskTimeouts.append(taskId, runnable);

    handler.postDelayed(runnable, timeout);
  }

  private void removeTimeout(int taskId) {
    Runnable timeout = taskTimeouts.get(taskId);

    if (timeout != null) {
      handler.removeCallbacks(timeout);
      taskTimeouts.remove(taskId);
    }
  }
}