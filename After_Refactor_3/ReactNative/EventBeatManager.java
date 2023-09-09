package com.facebook.react.fabric.events;

import androidx.annotation.NonNull;
import com.facebook.jni.HybridData;
import com.facebook.proguard.annotations.DoNotStrip;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.uimanager.events.BatchEventDispatchedListener;

/**
 * EventBeatManager acts as a proxy between the list of EventBeats registered in C++ and the Android side.
 */
public final class EventBeatManager implements BatchEventDispatchedListener {

    @DoNotStrip
    private final HybridData hybridData;

    /**
     * Initializes the hybrid data used to connect to native code.
     */
    private static native HybridData initHybrid();

    /**
     * Notifies native code that an event has happened.
     */
    private native void notifyNative();

    /**
     * Constructs an EventBeatManager instance.
     *
     * @deprecated Use the no-argument constructor instead.
     */
    @Deprecated(forRemoval = true, since = "Deprecated on v0.72.0")
    public EventBeatManager(@NonNull ReactApplicationContext reactApplicationContext) {
        this();
    }

    /**
     * Constructs an EventBeatManager instance.
     */
    public EventBeatManager() {
        this.hybridData = initHybrid();
    }

    /**
     * Called when a batch of events has been dispatched.
     * Notifies native code.
     */
    @Override
    public void onBatchEventDispatched() {
        notifyNative();
    }
}