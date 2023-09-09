package com.facebook.react.bridgeless.internal.bolts;

import java.io.Closeable;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

public class CancellationTokenSource implements Closeable {

  private final Object lock = new Object();
  private final List<CancellationTokenRegistration> registrations = new ArrayList<>();
  private final ScheduledExecutorService executor = BoltsExecutors.scheduled();
  private ScheduledFuture<?> scheduledCancellation;
  private boolean cancellationRequested;
  private boolean closed;

  private static final long IMMEDIATE = 0;
  private static final long CANCELLED = -1;

  public CancellationTokenSource() {}

  public boolean isCancellationRequested() {
    synchronized (lock) {
      checkIsNotClosed();
      return cancellationRequested;
    }
  }

  public CancellationToken getToken() {
    synchronized (lock) {
      checkIsNotClosed();
      return new CancellationToken(this);
    }
  }

  public void cancel() {
    synchronized (lock) {
      checkIsNotClosed();
      if (cancellationRequested) {
        return;
      }

      cancelScheduledCancellation();

      cancellationRequested = true;
      notifyListeners(new ArrayList<>(registrations));
    }
  }

  public void cancelAfter(final long delay) {
    cancelAfter(delay, TimeUnit.MILLISECONDS);
  }

  private void cancelAfter(long delay, TimeUnit timeUnit) {
    if (delay < CANCELLED) {
      throw new IllegalArgumentException("Delay must be >= " + CANCELLED);
    }

    synchronized (lock) {
      checkIsNotClosed();
      if (cancellationRequested) {
        return;
      }

      cancelScheduledCancellation();

      if (delay == IMMEDIATE) {
        cancellationRequested = true;
        notifyListeners(new ArrayList<>(registrations));
      } else if (delay != CANCELLED) {
        scheduledCancellation =
            executor.schedule(
                () -> {
                  synchronized (lock) {
                    scheduledCancellation = null;
                  }
                  cancel();
                },
                delay,
                timeUnit);
      }
    }
  }

  @Override
  public void close() {
    synchronized (lock) {
      if (closed) {
        return;
      }

      cancelScheduledCancellation();

      for (CancellationTokenRegistration registration : registrations) {
        registration.close();
      }
      registrations.clear();
      closed = true;
    }
  }

  /* package */ CancellationTokenRegistration register(Runnable action) {
    CancellationTokenRegistration registration;
    synchronized (lock) {
      checkIsNotClosed();

      registration = new CancellationTokenRegistration(this, action);
      if (cancellationRequested) {
        registration.runAction();
      } else {
        registrations.add(registration);
      }
    }
    return registration;
  }

  /* package */ void throwIfCancellationRequested() throws CancellationException {
    synchronized (lock) {
      checkIsNotClosed();
      if (cancellationRequested) {
        throw new CancellationException();
      }
    }
  }

  /* package */ void unregister(CancellationTokenRegistration registration) {
    synchronized (lock) {
      checkIsNotClosed();
      registrations.remove(registration);
    }
  }

  private void notifyListeners(List<CancellationTokenRegistration> registrations) {
    registrations.forEach(CancellationTokenRegistration::runAction);
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

  // Private helper methods

  private void cancelScheduledCancellation() {
    if (scheduledCancellation != null) {
      scheduledCancellation.cancel(true);
      scheduledCancellation = null;
    }
  }

  private void checkIsNotClosed() {
    if (closed) {
      throw new IllegalStateException("Object already closed");
    }
  }
}