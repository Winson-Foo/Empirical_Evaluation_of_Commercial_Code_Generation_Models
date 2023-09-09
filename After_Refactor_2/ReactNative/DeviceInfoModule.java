package com.facebook.react.modules.deviceinfo;

import android.content.Context;
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
import java.util.Optional;

@ReactModule(name = NativeDeviceInfoSpec.NAME)
public class DeviceInfoModule extends NativeDeviceInfoSpec implements LifecycleEventListener {
  private final ReactApplicationContext context;
  private float fontScale;
  private Optional<ReadableMap> previousDisplayMetrics = Optional.empty();

  public DeviceInfoModule(ReactApplicationContext context) {
    super(context);
    this.context = context;
    DisplayMetricsHolder.initDisplayMetricsIfNotInitialized(context);
    this.fontScale = context.getResources().getConfiguration().fontScale;
    context.addLifecycleEventListener(this);
  }

  public DeviceInfoModule(Context context) {
    super(null);
    this.context = null;
    DisplayMetricsHolder.initDisplayMetricsIfNotInitialized(context);
    this.fontScale = context.getResources().getConfiguration().fontScale;
  }

  public void emitUpdateDimensionsEvent() {
    if (context == null || !context.hasActiveReactInstance()) {
      return;
    }
    var displayMetrics = DisplayMetricsHolder.getDisplayMetricsWritableMap(fontScale);
    if (previousDisplayMetrics.isPresent() && displayMetrics.equals(previousDisplayMetrics.get())) {
      return;
    }
    previousDisplayMetrics = Optional.of(displayMetrics.copy());
    context.emitDeviceEvent("didUpdateDimensions", displayMetrics);
  }

  @Override
  public Map<String, Object> getTypedExportedConstants() {
    var displayMetrics = DisplayMetricsHolder.getDisplayMetricsWritableMap(fontScale);
    previousDisplayMetrics = Optional.of(displayMetrics.copy());
    var constants = new HashMap<String, Object>();
    constants.put("Dimensions", displayMetrics.toHashMap());
    return constants;
  }

  @Override
  public void onHostResume() {
    if (context == null) {
      return;
    }
    var fontScale = context.getResources().getConfiguration().fontScale;
    if (this.fontScale != fontScale) {
      this.fontScale = fontScale;
      emitUpdateDimensionsEvent();
    }
  }

  @Override
  public void onHostPause() {
  }

  @Override
  public void onHostDestroy() {
  }

  @Override
  public void invalidate() {
    super.invalidate();
    context.removeLifecycleEventListener(this);
  }
}