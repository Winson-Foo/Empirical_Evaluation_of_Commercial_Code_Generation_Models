/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.modules.deviceinfo;

import android.content.Context;

import androidx.annotation.Nullable;

import com.facebook.fbreact.specs.NativeDeviceInfoSpec;
import com.facebook.react.bridge.LifecycleEventListener;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactNoCrashSoftException;
import com.facebook.react.bridge.ReactSoftExceptionLogger;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.module.annotations.ReactModule;
import com.facebook.react.uimanager.DisplayMetricsHolder;

import java.util.HashMap;
import java.util.Map;

/**
 * Module that exposes Android Constants to JS.
 */
@ReactModule(name = NativeDeviceInfoSpec.NAME)
public class DeviceInfoModule extends NativeDeviceInfoSpec implements LifecycleEventListener {

  private final @Nullable ReactApplicationContext context;
  private final DisplayMetricsHolder displayMetricsHolder;
  private float fontScale;
  private @Nullable ReadableMap previousDisplayMetrics;

  public DeviceInfoModule(ReactApplicationContext context) {
    super(context);
    displayMetricsHolder = new DisplayMetricsHolder(context);
    this.context = context;
    init();
  }

  public DeviceInfoModule(Context context) {
    super(null);
    displayMetricsHolder = new DisplayMetricsHolder(context);
    this.context = null;
    init();
  }

  private void init() {
    displayMetricsHolder.initDisplayMetricsIfNotInitialized();
    fontScale = getFontScale();
    if (context != null) {
      context.addLifecycleEventListener(this);
    }
  }

  private float getFontScale() {
    return context != null ? context.getResources().getConfiguration().fontScale
            : displayMetricsHolder.getContext().getResources().getConfiguration().fontScale;
  }

  @Nullable
  @Override
  public Map<String, Object> getTypedExportedConstants() {
    WritableMap displayMetrics = displayMetricsHolder.getDisplayMetricsWritableMap(fontScale);

    // Cache the initial dimensions for later comparison in emitUpdateDimensionsEvent
    previousDisplayMetrics = displayMetrics.copy();

    HashMap<String, Object> constants = new HashMap<>();
    constants.put(Constants.DIMENSIONS, displayMetrics.toHashMap());
    return constants;
  }

  @Override
  public void onHostResume() {
    if (context == null) {
      return;
    }

    float currentFontScale = getFontScale();
    if (fontScale != currentFontScale) {
      fontScale = currentFontScale;
      emitUpdateDimensionsEvent();
    }
  }

  @Override
  public void onHostPause() {}

  @Override
  public void onHostDestroy() {}

  public void emitUpdateDimensionsEvent() {
    if (context == null || !context.hasActiveReactInstance()) {
      return;
    }

    try {
      WritableMap currentDisplayMetrics = displayMetricsHolder.getDisplayMetricsWritableMap(fontScale);
      if (previousDisplayMetrics == null) {
        previousDisplayMetrics = currentDisplayMetrics.copy();
      } else if (!currentDisplayMetrics.equals(previousDisplayMetrics)) {
        previousDisplayMetrics = currentDisplayMetrics.copy();
        context.emitDeviceEvent(Constants.DID_UPDATE_DIMENSIONS_EVENT, currentDisplayMetrics);
      }
    } catch (Exception e) {
      ReactSoftExceptionLogger.logSoftException(
          NAME, new ReactNoCrashSoftException("Could not emit update dimensions event", e));
    }
  }

  @Override
  public void invalidate() {
    super.invalidate();
    if (context != null) {
      context.removeLifecycleEventListener(this);
    }
  }

  private static class Constants {
    public static final String DIMENSIONS = "Dimensions";
    public static final String DID_UPDATE_DIMENSIONS_EVENT = "didUpdateDimensions";
  }
}

