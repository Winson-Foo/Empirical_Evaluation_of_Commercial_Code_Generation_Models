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

@ReactModule(name = DeviceInfoModule.MODULE_NAME)
public class DeviceInfoModule extends NativeDeviceInfoSpec implements LifecycleEventListener {
    public static final String MODULE_NAME = "DeviceInfo";
    public static final String DID_UPDATE_DIMENSIONS_EVENT_NAME = "didUpdateDimensions";
    public static final String CANNOT_EMIT_UPDATE_DIMENSIONS_EXCEPTION_MESSAGE = "No active CatalystInstance, cannot emitUpdateDimensionsEvent";

    private final ReactApplicationContext reactApplicationContext;
    private final DisplayMetricsHolder displayMetricsHolder;
    private final ReactSoftExceptionLogger reactSoftExceptionLogger;

    private final float fontScale;
    private ReadableMap previousDisplayMetrics;

    public DeviceInfoModule(ReactApplicationContext context) {
        super(context);
        this.reactApplicationContext = context;
        this.fontScale = context.getResources().getConfiguration().fontScale;
        this.displayMetricsHolder = DisplayMetricsHolder.getInstance();
        this.reactSoftExceptionLogger = ReactSoftExceptionLogger.getInstance();
        this.reactApplicationContext.addLifecycleEventListener(this);
    }

    public DeviceInfoModule(Context context) {
        super(null);
        this.reactApplicationContext = null;
        this.fontScale = context.getResources().getConfiguration().fontScale;
        this.displayMetricsHolder = DisplayMetricsHolder.getInstance();
        this.reactSoftExceptionLogger = ReactSoftExceptionLogger.getInstance();
    }

    @Override
    public Map<String, Object> getTypedExportedConstants() {
        WritableMap displayMetrics = displayMetricsHolder.getDisplayMetricsWritableMap(fontScale);
        previousDisplayMetrics = previousDisplayMetrics == null ? displayMetrics.copy() : previousDisplayMetrics;
        HashMap<String, Object> constants = new HashMap<>();
        constants.put("Dimensions", displayMetrics.toHashMap());
        return constants;
    }

    @Override
    public void onHostResume() {
        if (reactApplicationContext == null) {
            return;
        }
        float currentFontScale = reactApplicationContext.getResources().getConfiguration().fontScale;
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
        if (reactApplicationContext == null) {
            return;
        }
        if (reactApplicationContext.hasActiveReactInstance()) {
            WritableMap displayMetrics = displayMetricsHolder.getDisplayMetricsWritableMap(fontScale);
            if (previousDisplayMetrics == null || !displayMetrics.equals(previousDisplayMetrics)) {
                previousDisplayMetrics = displayMetrics.copy();
                reactApplicationContext.emitDeviceEvent(DID_UPDATE_DIMENSIONS_EVENT_NAME, displayMetrics);
            }
        } else {
            String module = MODULE_NAME;
            ReactNoCrashSoftException exception = new ReactNoCrashSoftException(CANNOT_EMIT_UPDATE_DIMENSIONS_EXCEPTION_MESSAGE);
            reactSoftExceptionLogger.logSoftException(module, exception);
        }
    }

    @Override
    public void invalidate() {
        super.invalidate();
        if (reactApplicationContext != null) {
            reactApplicationContext.removeLifecycleEventListener(this);
        }
    }
}