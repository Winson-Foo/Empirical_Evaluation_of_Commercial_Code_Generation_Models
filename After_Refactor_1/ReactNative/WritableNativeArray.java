package com.facebook.react.bridge;

import androidx.annotation.Nullable;
import com.facebook.infer.annotation.Assertions;
import com.facebook.jni.HybridData;
import com.facebook.proguard.annotations.DoNotStrip;

@DoNotStrip
public class WritableNativeArray extends ReadableNativeArray implements WritableArray {

    private static final int EMPTY_ARRAY_SIZE = 0;

    static {
        ReactBridge.staticInit();
    }

    public WritableNativeArray() {
        super(initHybrid());
    }

    // Push null value to the array
    @Override
    public native void pushNull();

    // Push boolean value to the array
    @Override
    public native void pushBoolean(boolean value);

    // Push double value to the array
    @Override
    public native void pushDouble(double value);

    // Push integer value to the array
    @Override
    public native void pushInt(int value);

    // Push string value to the array
    @Override
    public native void pushString(@Nullable String value);

    // Push a ReadableArray to the array
    // Note: this consumes the array so do not reuse it.
    @Override
    public void pushArray(@Nullable ReadableArray array) {
        Assertions.assertCondition(
                array == null || array instanceof ReadableNativeArray, "Illegal type provided");
        pushNativeArray((ReadableNativeArray) array);
    }

    // Push a ReadableMap to the map
    // Note: this consumes the map so do not reuse it.
    @Override
    public void pushMap(@Nullable ReadableMap map) {
        Assertions.assertCondition(
                map == null || map instanceof ReadableNativeMap, "Illegal type provided");
        pushNativeMap((ReadableNativeMap) map);
    }

    // Initialize the hybrid data with an empty array
    private static native HybridData initHybrid();

    // Push a ReadableArray to the array
    private native void pushNativeArray(ReadableNativeArray array);

    // Push a ReadableMap to the map
    private native void pushNativeMap(ReadableNativeMap map);

    // Get the size of the array. Override the implementation in super class.
    @Override
    public int size() {
        return isEmpty() ? EMPTY_ARRAY_SIZE : getArraySize();
    }

    // Check if the array is empty
    private boolean isEmpty() {
        return getArraySize() == 0;
    }
}