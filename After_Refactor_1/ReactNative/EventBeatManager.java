package com.facebook.react.fabric.events;

import android.annotation.SuppressLint;
import androidx.annotation.NonNull;
import com.facebook.jni.HybridData;
import com.facebook.proguard.annotations.DoNotStrip;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.fabric.FabricSoLoader;
import com.facebook.react.uimanager.events.BatchEventDispatchedListener;

/**
 * Manages the dispatch of events from C++ to the Android side.
 */
@SuppressLint("MissingNativeLoadLibrary")
public final class EventBeatManager implements BatchEventDispatchedListener {

  static {
    FabricSoLoader.staticInit();
  }

  @DoNotStrip private final HybridData mHybridData;

  /**
   * Creates a new instance of EventBeatManager.
   */
  public EventBeatManager() {
    mHybridData = initHybrid();
  }

  /**
   * Creates a new instance of EventBeatManager with the given ReactApplicationContext.
   *
   * @param reactApplicationContext The context used to create the instance.
   */
  public EventBeatManager(@NonNull ReactApplicationContext reactApplicationContext) {
    this();
  }

  /**
   * Dispatches events from C++ to the Android side.
   */
  private native void dispatchEvents();

  @Override
  public void onBatchEventDispatched() {
    dispatchEvents();
  }

  private static native HybridData initHybrid();
}