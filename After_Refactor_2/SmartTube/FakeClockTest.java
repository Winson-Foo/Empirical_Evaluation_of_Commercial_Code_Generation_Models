package com.google.android.exoplayer2.testutil;

import static com.google.common.truth.Truth.assertThat;

import android.os.ConditionVariable;
import android.os.HandlerThread;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.android.exoplayer2.util.Clock;
import com.google.android.exoplayer2.util.HandlerWrapper;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.annotation.Config;

@RunWith(AndroidJUnit4.class)
@Config(shadows = {RobolectricUtil.CustomLooper.class, RobolectricUtil.CustomMessageQueue.class})
public final class FakeClockTest {
  private static final long TIMEOUT_MS = 10000;
  private static final long INITIAL_TIME = 0;
  private static final long MODERATE_DELAY = 100;
  private static final long LONG_DELAY = 200;
  private static final long VERY_LONG_DELAY = 300;
  private static final long EXTREME_DELAY = 400;
  private static final long SHORT_SLEEP = 1000;

  @Test
  public void testAdvanceTime() {
    FakeClock fakeClock = new FakeClock(INITIAL_TIME);
    assertThat(fakeClock.elapsedRealtime()).isEqualTo(INITIAL_TIME);
    fakeClock.advanceTime(MODERATE_DELAY);
    assertThat(fakeClock.elapsedRealtime()).isEqualTo(MODERATE_DELAY);
    fakeClock.advanceTime(0);
    assertThat(fakeClock.elapsedRealtime()).isEqualTo(MODERATE_DELAY);
  }

  @Test
  public void testSleep() throws InterruptedException {
    FakeClock fakeClock = new FakeClock(INITIAL_TIME);
    SleeperThread sleeperThread1 = new SleeperThread(fakeClock, SHORT_SLEEP);
    SleeperThread sleeperThread2 = new SleeperThread(fakeClock, INITIAL_TIME);
    SleeperThread sleeperThread3 = new SleeperThread(fakeClock, LONG_DELAY);
    SleeperThread sleeperThread4 = new SleeperThread(fakeClock, VERY_LONG_DELAY);
    SleeperThread sleeperThread5 = new SleeperThread(fakeClock, EXTREME_DELAY);

    assertSleeping(sleeperThread1, TIMEOUT_MS, true);
    assertSleeping(sleeperThread2, 0, false);
    assertAllSleeping(sleeperThread1, sleeperThread3, sleeperThread4, sleeperThread5);
    assertAwake(sleeperThread1, sleeperThread3);
    assertAllSleeping(sleeperThread4, sleeperThread5);
    assertAllAwake(sleeperThread1, sleeperThread3, sleeperThread4, sleeperThread5);
  }

  @Test
  public void testPostDelayed() {
    HandlerThread handlerThread = new HandlerThread("FakeClockTest thread");
    handlerThread.start();
    FakeClock fakeClock = new FakeClock(INITIAL_TIME);
    HandlerWrapper handler = fakeClock.createHandler(handlerThread.getLooper(), null);

    TestRunnable testRunnable1 = new TestRunnable();
    TestRunnable testRunnable2 = new TestRunnable();
    TestRunnable testRunnable3 = new TestRunnable();
    TestRunnable testRunnable4 = new TestRunnable();
    TestRunnable testRunnable5 = new TestRunnable();

    handler.postDelayed(testRunnable1, INITIAL_TIME);
    handler.postDelayed(testRunnable2, MODERATE_DELAY);
    handler.postDelayed(testRunnable3, LONG_DELAY);
    waitForHandler(handler);
    assertHasRun(testRunnable1, true);
    assertHasRun(testRunnable2, false);
    assertHasRun(testRunnable3, false);

    handler.postDelayed(testRunnable4, MODERATE_DELAY - INITIAL_TIME);
    handler.postDelayed(testRunnable5, LONG_DELAY - INITIAL_TIME);
    waitForHandler(handler);
    assertHasRun(testRunnable1, true);
    assertHasRun(testRunnable2, true);
    assertHasRun(testRunnable3, false);
    assertHasRun(testRunnable4, true);
    assertHasRun(testRunnable5, false);

    fakeClock.advanceTime(VERY_LONG_DELAY - MODERATE_DELAY);
    waitForHandler(handler);
    assertHasRun(testRunnable1, true);
    assertHasRun(testRunnable2, true);
    assertHasRun(testRunnable3, false);
    assertHasRun(testRunnable4, true);
    assertHasRun(testRunnable5, true);

    fakeClock.advanceTime(EXTREME_DELAY - VERY_LONG_DELAY);
    waitForHandler(handler);
    assertHasRun(testRunnable1, true);
    assertHasRun(testRunnable2, true);
    assertHasRun(testRunnable3, true);
    assertHasRun(testRunnable4, true);
    assertHasRun(testRunnable5, true);
  }

  private void assertSleeping(SleeperThread sleeperThread, long timeout, boolean expected) {
    try {
      assertThat(sleeperThread.waitUntilAsleep(timeout)).isEqualTo(expected);
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
    assertThat(sleeperThread.isSleeping()).isEqualTo(expected);
  }

  private void assertAllSleeping(SleeperThread... sleeperThreads) {
    for (SleeperThread sleeperThread : sleeperThreads) {
      assertThat(sleeperThread.waitUntilAsleep(TIMEOUT_MS)).isTrue();
      assertThat(sleeperThread.isSleeping()).isTrue();
    }
  }

  private void assertAwake(SleeperThread sleeperThread1, SleeperThread sleeperThread2) {
    fakeClock.advanceTime(LONG_DELAY - MODERATE_DELAY);
    waitForHandler(handler);
    assertThat(sleeperThread1.waitUntilAwake(TIMEOUT_MS)).isTrue();
    assertThat(sleeperThread2.waitUntilAwake(TIMEOUT_MS)).isTrue();
    assertThat(sleeperThread1.isSleeping()).isFalse();
    assertThat(sleeperThread2.isSleeping()).isFalse();
  }

  private void assertAllAwake(SleeperThread... sleeperThreads) {
    fakeClock.advanceTime(SHORT_SLEEP + LONG_DELAY + VERY_LONG_DELAY + EXTREME_DELAY);
    for (SleeperThread sleeperThread : sleeperThreads) {
      assertThat(sleeperThread.waitUntilAwake(TIMEOUT_MS)).isTrue();
      assertThat(sleeperThread.isSleeping()).isFalse();
    }
  }

  private void assertHasRun(TestRunnable testRunnable, boolean expected) {
    assertThat(testRunnable.hasRun).isEqualTo(expected);
  }

  private void waitForHandler(HandlerWrapper handler) {
    final ConditionVariable handlerFinished = new ConditionVariable();
    handler.post(handlerFinished::open);
    handlerFinished.block();
  }

  private static final class SleeperThread extends Thread {
    private final Clock clock;
    private final long sleepDurationMs;
    private final CountDownLatch fallAsleepCountDownLatch;
    private final CountDownLatch wakeUpCountDownLatch;

    private volatile boolean isSleeping;

    public SleeperThread(Clock clock, long sleepDurationMs) {
      this.clock = clock;
      this.sleepDurationMs = sleepDurationMs;
      this.fallAsleepCountDownLatch = new CountDownLatch(1);
      this.wakeUpCountDownLatch = new CountDownLatch(1);
    }

    public boolean waitUntilAsleep(long timeoutMs) throws InterruptedException {
      return fallAsleepCountDownLatch.await(timeoutMs, TimeUnit.MILLISECONDS);
    }

    public boolean waitUntilAwake(long timeoutMs) throws InterruptedException {
      return wakeUpCountDownLatch.await(timeoutMs, TimeUnit.MILLISECONDS);
    }

    public boolean isSleeping() {
      return isSleeping;
    }

    @Override
    public void run() {
      // This relies on the FakeClock's methods synchronizing on its own monitor to ensure that
      // any interactions with it occur only after sleep() has called wait() or returned.
      synchronized (clock) {
        isSleeping = true;
        fallAsleepCountDownLatch.countDown();
        clock.sleep(sleepDurationMs);
        isSleeping = false;
        wakeUpCountDownLatch.countDown();
      }
    }
  }

  private static final class TestRunnable implements Runnable {
    public boolean hasRun;

    @Override
    public void run() {
      hasRun = true;
    }
  }
} 