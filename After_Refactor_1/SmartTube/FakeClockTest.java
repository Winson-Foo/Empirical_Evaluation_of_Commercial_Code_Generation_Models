package com.google.android.exoplayer2.testutil;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import android.os.ConditionVariable;
import android.os.HandlerThread;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/** Unit tests for {@link FakeClock}. */
@RunWith(AndroidJUnit4.class)
public final class FakeClockUnitTests {

    private static final long TIMEOUT_MS = 10000;

    private FakeClock fakeClock;
    private HandlerThread handlerThread;

    /** Sets up the test environment. */
    @Before
    public void setUp() {
        fakeClock = new FakeClock(0);
        handlerThread = new HandlerThread("FakeClockTest thread");
        handlerThread.start();
    }

    /** Cleans up the test environment. */
    @After
    public void tearDown() {
        handlerThread.quit();
    }

    /** Tests the {@link FakeClock#advanceTime(long)} method. */
    @Test
    public void testAdvanceTime() {
        fakeClock.advanceTime(500);
        assertTrue(500 == fakeClock.elapsedRealtime());

        fakeClock.advanceTime(0);
        assertTrue(500 == fakeClock.elapsedRealtime());

        fakeClock.advanceTime(-500);
        assertTrue(0 == fakeClock.elapsedRealtime());
    }

    /** Tests the {@link FakeClock#sleep(long)} method. */
    @Test
    public void testSleep() throws InterruptedException {
        SleeperThread sleeperThread = new SleeperThread(fakeClock, 1000);
        sleeperThread.start();
        assertTrue(sleeperThread.waitUntilAsleep(TIMEOUT_MS));
        assertTrue(sleeperThread.isSleeping());

        fakeClock.advanceTime(1000);
        sleeperThread.join(TIMEOUT_MS);
        assertFalse(sleeperThread.isSleeping());

        sleeperThread = new SleeperThread(fakeClock, 0);
        sleeperThread.start();
        sleeperThread.join();
        assertFalse(sleeperThread.isSleeping());

        SleeperThread[] sleeperThreads = new SleeperThread[5];
        sleeperThreads[0] = new SleeperThread(fakeClock, 1000);
        sleeperThreads[1] = new SleeperThread(fakeClock, 1000);
        sleeperThreads[2] = new SleeperThread(fakeClock, 2000);
        sleeperThreads[3] = new SleeperThread(fakeClock, 3000);
        sleeperThreads[4] = new SleeperThread(fakeClock, 4000);
        for (SleeperThread thread : sleeperThreads) {
            thread.start();
            assertTrue(thread.waitUntilAsleep(TIMEOUT_MS));
        }

        assertSleepingStates(new boolean[] {true, true, true, true, true}, sleeperThreads);

        fakeClock.advanceTime(1500);
        assertTrue(sleeperThreads[0].waitUntilAwake(TIMEOUT_MS));
        assertTrue(sleeperThreads[1].waitUntilAwake(TIMEOUT_MS));
        assertSleepingStates(new boolean[] {false, false, true, true, true}, sleeperThreads);

        fakeClock.advanceTime(2000);
        assertTrue(sleeperThreads[2].waitUntilAwake(TIMEOUT_MS));
        assertTrue(sleeperThreads[3].waitUntilAwake(TIMEOUT_MS));
        assertSleepingStates(new boolean[] {false, false, false, false, true}, sleeperThreads);

        fakeClock.advanceTime(2000);
        for (SleeperThread thread : sleeperThreads) {
            thread.join(TIMEOUT_MS);
        }
        assertSleepingStates(new boolean[] {false, false, false, false, false}, sleeperThreads);
    }

    /** Tests the {@link FakeClock#createHandler} method. */
    @Test
    public void testCreateHandler() throws InterruptedException {
        HandlerWrapper handler = fakeClock.createHandler(handlerThread.getLooper(), null);

        TestRunnable[] testRunnables = {
                new TestRunnable(),
                new TestRunnable(),
                new TestRunnable(),
                new TestRunnable(),
                new TestRunnable()
        };
        handler.postDelayed(testRunnables[0], 0);
        handler.postDelayed(testRunnables[1], 100);
        handler.postDelayed(testRunnables[2], 200);

        waitForHandler(handler);
        assertTestRunnableStates(new boolean[] {true, false, false, false, false}, testRunnables);

        fakeClock.advanceTime(150);
        handler.postDelayed(testRunnables[3], 50);

        waitForHandler(handler);
        assertTestRunnableStates(new boolean[] {true, true, false, false, false}, testRunnables);

        fakeClock.advanceTime(50);
        waitForHandler(handler);
        assertTestRunnableStates(new boolean[] {true, true, true, true, false}, testRunnables);

        fakeClock.advanceTime(1000);
        waitForHandler(handler);
        assertTestRunnableStates(new boolean[] {true, true, true, true, true}, testRunnables);
    }

    /**
     * Verifies that an array of SleeperThreads has the expected sleeping states.
     *
     * @param states The expected sleeping states.
     * @param sleeperThreads The SleeperThreads to verify.
     */
    private static void assertSleepingStates(boolean[] states, SleeperThread[] sleeperThreads) {
        for (int i = 0; i < sleeperThreads.length; i++) {
            assertTrue(sleeperThreads[i].isSleeping() == states[i]);
        }
    }

    /**
     * Waits for the handler to finish.
     *
     * @param handlerWrapper The handler to wait for.
     */
    private static void waitForHandler(HandlerWrapper handlerWrapper) {
        final ConditionVariable handlerFinished = new ConditionVariable();
        handlerWrapper.post(handlerFinished::open);
        handlerFinished.block();
    }

    /**
     * Verifies that an array of TestRunnables have been run as expected.
     *
     * @param states The expected run states.
     * @param testRunnables The TestRunnables to verify.
     */
    private static void assertTestRunnableStates(boolean[] states, TestRunnable[] testRunnables) {
        for (int i = 0; i < testRunnables.length; i++) {
            assertTrue(testRunnables[i].hasRun == states[i]);
        }
    }

    /**
     * Runs a {@link Clock#sleep(long)} method in a separate thread.
     */
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

        /**
         * Waits for the sleeper thread to fall asleep.
         *
         * @param timeoutMs The maximum amount of time to wait. If the thread hasn't fallen asleep
         *     by this time, the method returns false.
         */
        public boolean waitUntilAsleep(long timeoutMs) throws InterruptedException {
            return fallAsleepCountDownLatch.await(timeoutMs, TimeUnit.MILLISECONDS);
        }

        /**
         * Waits for the sleeper thread to wake up.
         *
         * @param timeoutMs The maximum amount of time to wait. If the thread hasn't woken up by
         *     this time, the method returns false.
         */
        public boolean waitUntilAwake(long timeoutMs) throws InterruptedException {
            return wakeUpCountDownLatch.await(timeoutMs, TimeUnit.MILLISECONDS);
        }

        /** Returns true if the thread is currently sleeping, false otherwise. */
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

    /** A simple Runnable that sets a boolean flag when it's been run. */
    private static final class TestRunnable implements Runnable {

        public boolean hasRun;

        @Override
        public void run() {
            hasRun = true;
        }
    }
}