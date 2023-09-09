package com.facebook.react.bridgeless.internal.bolts;

import java.io.Closeable;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

public class CancellationTokenSource implements Closeable {

  private final Object mutex = new Object();
  private final List<CancellationTokenRegistration> registrations = new ArrayList<>();
  private final BoltsScheduledExecutorService executor = new BoltsScheduledExecutorService();
  private ScheduledFuture<?> scheduledCancellation;
  private boolean isCancelled;
  private boolean closed;

  public static final long CANCEL_IMMEDIATELY = 0;
  public static final long CANCEL_NOT_SCHEDULED = -1;

  public CancellationTokenSource() {}

  public boolean isCancellationRequested() {
    synchronized (mutex) {
      checkNotClosed();
      return isCancelled;
    }
  }

  public CancellationToken getToken() {
    synchronized (mutex) {
      checkNotClosed();
      return new CancellationToken(this);
    }
  }

  public void cancel() {
    List<CancellationTokenRegistration> registrations;
    synchronized (mutex) {
      checkNotClosed();
      if (isCancelled) {
        return;
      }

      cancelScheduledCancellation();

      isCancelled = true;
      registrations = new ArrayList<>(this.registrations);
    }
    notifyListeners(registrations);
  }

  public void cancelAfter(final long delay) {
    if (delay < CANCEL_NOT_SCHEDULED) {
      throw new IllegalArgumentException("Delay must be >= -1");
    }

    synchronized (mutex) {
      checkNotClosed();
      if (isCancelled) {
        return;
      }

      cancelScheduledCancellation();

      if (delay == CANCEL_IMMEDIATELY) {
        cancel();
      } else if (delay != CANCEL_NOT_SCHEDULED) {
        scheduledCancellation =
            executor.schedule(
                () -> {
                  synchronized (mutex) {
                    scheduledCancellation = null;
                  }
                  cancel();
                },
                delay,
                TimeUnit.MILLISECONDS);
      }
    }
  }

  @Override
  public void close() {
    synchronized (mutex) {
      if (closed) {
        return;
      }

      cancelScheduledCancellation();

      try (BoltsScheduledExecutorService executor = this.executor) {
        for (CancellationTokenRegistration registration : registrations) {
          registration.close();
        }
      }
      registrations.clear();
      closed = true;
    }
  }

  CancellationTokenRegistration register(Runnable action) {
    CancellationTokenRegistration registration;
    synchronized (mutex) {
      checkNotClosed();

      registration = new CancellationTokenRegistration(this, action);
      if (isCancelled) {
        registration.runAction();
      } else {
        registrations.add(registration);
      }
    }
    return registration;
  }

  void unregister(CancellationTokenRegistration registration) {
    synchronized (mutex) {
      checkNotClosed();
      registrations.remove(registration);
    }
  }

  private void notifyListeners(List<CancellationTokenRegistration> registrations) {
    for (CancellationTokenRegistration registration : registrations) {
      registration.runAction();
    }
  }

  @Override
  public String toString() {
    return String.format(
        Locale.US,
        "%s@%s[cancellationRequested=%s]",
        getClass().getName(),
        Integer.toHexString(hashCode()),
        Boolean.toString(isCancellationRequested()));
  }

  private void cancelScheduledCancellation() {
    if (scheduledCancellation != null) {
      scheduledCancellation.cancel(true);
      scheduledCancellation = null;
    }
  }

  private void checkNotClosed() {
    if (closed) {
      throw new IllegalStateException("Object already closed");
    }
  }
}

public class CancellationToken {

  private final CancellationTokenSource source;

  CancellationToken(CancellationTokenSource source) {
    this.source = source;
  }

  public boolean isCancellationRequested() {
    return source.isCancellationRequested();
  }

  public void throwIfCancellationRequested() throws CancellationException {
    if (isCancellationRequested()) {
      throw new CancellationException();
    }
  }

  public CancellationTokenRegistration register(Runnable action) {
    return source.register(action);
  }
}

public class CancellationTokenRegistration implements Closeable {

  private final CancellationTokenSource source;
  private final Runnable action;

  CancellationTokenRegistration(CancellationTokenSource source, Runnable action) {
    this.source = source;
    this.action = action;
  }

  public void unregister() {
    source.unregister(this);
  }

  /* package */ void runAction() {
    action.run();
  }

  @Override
  public void close() {
    unregister();
  }
}

public class BoltsScheduledExecutorService implements ScheduledExecutorService, Closeable {

  private final ScheduledExecutorService executor = BoltsExecutors.scheduled();

  @Override
  public void close() {
    executor.shutdown();
  }

  // delegate other methods to executor
}