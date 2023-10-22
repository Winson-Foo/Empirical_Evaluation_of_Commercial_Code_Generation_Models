package com.facebook.react.bridgeless.internal.bolts;

import java.io.Closeable;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.logging.Logger;

public class CancellationTokenSource implements Closeable {

  private static final Logger LOGGER = Logger.getLogger(CancellationTokenSource.class.getName());

  private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
  private final List<CancellationTokenRegistration> registrations = new ArrayList<>();
  private final ScheduledExecutorService executor = BoltsExecutors.scheduled();
  private ScheduledFuture<?> scheduledCancellation;
  private boolean cancellationRequested;
  private boolean closed;

  public CancellationTokenSource() {}

  public boolean isCancellationRequested() {
    lock.readLock().lock();
    try {
      checkIfClosed();
      return cancellationRequested;
    } finally {
      lock.readLock().unlock();
    }
  }

  public CancellationToken getToken() {
    lock.readLock().lock();
    try {
      checkIfClosed();
      return new CancellationToken(this);
    } finally {
      lock.readLock().unlock();
    }
  }

  public void cancel() {
    List<CancellationTokenRegistration> registrations;
    lock.writeLock().lock();
    try {
      checkIfClosed();
      if (cancellationRequested) {
        LOGGER.warning("Cancellation already requested");
        return;
      }

      cancelScheduledCancellation();

      cancellationRequested = true;
      registrations = new ArrayList<>(this.registrations);
    } finally {
      lock.writeLock().unlock();
    }
    notifyListeners(registrations);
  }

  public void cancelAfter(final long delay) {
    cancelAfter(delay, TimeUnit.MILLISECONDS);
  }

  private void cancelAfter(long delay, TimeUnit timeUnit) {
    if (delay < -1) {
      LOGGER.warning("Delay must be >= -1");
      throw new IllegalArgumentException("Delay must be >= -1");
    }

    if (delay == 0) {
      cancel();
      return;
    }

    lock.writeLock().lock();
    try {
      if (cancellationRequested) {
        LOGGER.warning("Cancellation already requested");
        return;
      }

      cancelScheduledCancellation();

      if (delay != -1) {
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
    } finally {
      lock.writeLock().unlock();
    }
  }

  @Override
  public void close() {
    lock.writeLock().lock();
    try {
      if (closed) {
        LOGGER.warning("Object already closed");
        return;
      }

      cancelScheduledCancellation();

      List<CancellationTokenRegistration> registrations = new ArrayList<>(this.registrations);
      for (CancellationTokenRegistration registration : registrations) {
        registration.close();
      }
      this.registrations.clear();
      closed = true;
    } finally {
      lock.writeLock().unlock();
    }
  }

  /* package */ CancellationTokenRegistration register(Runnable action) {
    CancellationTokenRegistration ctr;
    lock.writeLock().lock();
    try {
      checkIfClosed();

      ctr = new CancellationTokenRegistration(this, action);
      if (cancellationRequested) {
        ctr.runAction();
      } else {
        registrations.add(ctr);
      }
    } finally {
      lock.writeLock().unlock();
    }
    return ctr;
  }

  /* package */ void throwIfCancellationRequested() throws CancellationException {
    lock.readLock().lock();
    try {
      checkIfClosed();
      if (cancellationRequested) {
        throw new CancellationException();
      }
    } finally {
      lock.readLock().unlock();
    }
  }

  /* package */ void unregister(CancellationTokenRegistration registration) {
    lock.writeLock().lock();
    try {
      checkIfClosed();
      registrations.remove(registration);
    } finally {
      lock.writeLock().unlock();
    }
  }

  private void notifyListeners(List<CancellationTokenRegistration> registrations) {
    lock.readLock().lock();
    try {
      for (CancellationTokenRegistration registration : registrations) {
        registration.runAction();
      }
    } finally {
      lock.readLock().unlock();
    }
  }

  @Override
  public String toString() {
    lock.readLock().lock();
    try {
      return String.format(
          Locale.US,
          "%s@%s[cancellationRequested=%s]",
          getClass().getName(),
          Integer.toHexString(hashCode()),
          Boolean.toString(isCancellationRequested()));
    } finally {
      lock.readLock().unlock();
    }
  }

  private void checkIfClosed() {
    if (closed) {
      LOGGER.warning("Object already closed");
      throw new IllegalStateException("Object already closed");
    }
  }

  private void cancelScheduledCancellation() {
    if (scheduledCancellation != null) {
      scheduledCancellation.cancel(true);
      scheduledCancellation = null;
    }
  }
}