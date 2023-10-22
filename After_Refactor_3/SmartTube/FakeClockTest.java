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

/** Unit test for {@link FakeClock}. */
@RunWith(AndroidJUnit4.class)
@Config(shadows = {RobolectricUtil.CustomLooper.class, RobolectricUtil.CustomMessageQueue.class})
public final class FakeClockTest {

    private static final long TIMEOUT_MS = 10000;
    private static final long FAKE_INITIAL_TIME_MS = 2000;
    private static final long MILLIS_500 = 500;
    private static final long MILLIS_1000 = 1000;
    private static final long MILLIS_1500 = 1500;
    private static final long MILLIS_2000 = 2000;
    private static final long MILLIS_3000 = 3000;
    private static final long MILLIS_4000 = 4000;
    private static final long ZERO = 0;

    @Test
    public void testAdvanceTime() {
        FakeClock fakeClock = new FakeClock(FAKE_INITIAL_TIME_MS);
        assertThat(fakeClock.elapsedRealtime()).isEqualTo(FAKE_INITIAL_TIME_MS);
        fakeClock.advanceTime(MILLIS_500);
        assertThat(fakeClock.elapsedRealtime()).isEqualTo(FAKE_INITIAL_TIME_MS + MILLIS_500);
        fakeClock.advanceTime(ZERO);
        assertThat(fakeClock.elapsedRealtime()).isEqualTo(FAKE_INITIAL_TIME_MS + MILLIS_500);
    }

    @Test
    public void testSleep() throws InterruptedException {
        FakeClock fakeClock = new FakeClock(ZERO);
        SleeperThread sleeperThread = new SleeperThread(fakeClock, MILLIS_1000);
        sleeperThread.start();
        assertThat(sleeperThread.waitUntilAsleep(TIMEOUT_MS)).isTrue();
        assertThat(sleeperThread.isSleeping()).isTrue();
        fakeClock.advanceTime(MILLIS_1000);
        sleeperThread.join(TIMEOUT_MS);
        assertThat(sleeperThread.isSleeping()).isFalse();

        sleeperThread = new SleeperThread(fakeClock, ZERO);
        sleeperThread.start();
        sleeperThread.join();
        assertThat(sleeperThread.isSleeping()).isFalse();

        SleeperThread[] sleeperThreads = new SleeperThread[5];
        sleeperThreads[0] = new SleeperThread(fakeClock, MILLIS_1000);
        sleeperThreads[1] = new SleeperThread(fakeClock, MILLIS_1000);
        sleeperThreads[2] = new SleeperThread(fakeClock, MILLIS_2000);
        sleeperThreads[3] = new SleeperThread(fakeClock, MILLIS_3000);
        sleeperThreads[4] = new SleeperThread(fakeClock, MILLIS_4000);
        for (SleeperThread thread : sleeperThreads) {
            thread.start();
            assertThat(thread.waitUntilAsleep(TIMEOUT_MS)).isTrue();
        }
        assertSleepingStates(new boolean[] {true, true, true, true, true}, sleeperThreads);
        fakeClock.advanceTime(MILLIS_1500);
        assertThat(sleeperThreads[0].waitUntilAwake(TIMEOUT_MS)).isTrue();
        assertThat(sleeperThreads[1].waitUntilAwake(TIMEOUT_MS)).isTrue();
        assertSleepingStates(new boolean[] {false, false, true, true, true}, sleeperThreads);
        fakeClock.advanceTime(MILLIS_2000);
        assertThat(sleeperThreads[2].waitUntilAwake(TIMEOUT_MS)).isTrue();
        assertThat(sleeperThreads[3].waitUntilAwake(TIMEOUT_MS)).isTrue();
        assertSleepingStates(new boolean[] {false, false, false, false, true}, sleeperThreads);
        fakeClock.advanceTime(MILLIS_2000);
        for (SleeperThread thread : sleeperThreads) {
            thread.join(TIMEOUT_MS);
        }
        assertSleepingStates(new boolean[] {false, false, false, false, false}, sleeperThreads);
    }

    private void assertSleepingStates(boolean[] states, SleeperThread[] sleeperThreads) {
        for (int i = 0; i < sleeperThreads.length; i++) {
            assertThat(sleeperThreads[i].isSleeping()).isEqualTo(states[i]);
        }
    }

    @Test
    public void testPostDelayed() {
        HandlerThread handlerThread = new HandlerThread("FakeClockTest thread");
        handlerThread.start();
        FakeClock fakeClock = new FakeClock(ZERO);
        HandlerWrapper handler = fakeClock.createHandler(handlerThread.getLooper(), null);

        TestRunnable[] testRunnables = {
            new TestRunnable(),
            new TestRunnable(),
            new TestRunnable(),
            new TestRunnable(),
            new TestRunnable()
        };
        handler.postDelayed(testRunnables[0], ZERO);
        handler.postDelayed(testRunnables[1], MILLIS_100);
        handler.postDelayed(testRunnables[2], MILLIS_200);
        waitForHandler(handler);
        assertTestRunnableStates(new boolean[] {true, false, false, false, false}, testRunnables);

        fakeClock.advanceTime(MILLIS_150);
        handler.postDelayed(testRunnables[3], MILLIS_50);
        handler.postDelayed(testRunnables[4], MILLIS_100);
        waitForHandler(handler);
        assertTestRunnableStates(new boolean[] {true, true, false, false, false}, testRunnables);

        fakeClock.advanceTime(MILLIS_50);
        waitForHandler(handler);
        assertTestRunnableStates(new boolean[] {true, true, true, true, false}, testRunnables);

        fakeClock.advanceTime(MILLIS_1000);
        waitForHandler(handler);
        assertTestRunnableStates(new boolean[] {true, true, true, true, true}, testRunnables);
    }

    private void waitForHandler(HandlerWrapper handler) {
        final ConditionVariable handlerFinished = new ConditionVariable();
        handler.post(handlerFinished::open);
        handlerFinished.block();
    }

    private void assertTestRunnableStates(boolean[] states, TestRunnable[] testRunnables) {
        for (int i = 0; i < testRunnables.length; i++) {
            assertThat(testRunnables[i].hasRun).isEqualTo(states[i]);
        }
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