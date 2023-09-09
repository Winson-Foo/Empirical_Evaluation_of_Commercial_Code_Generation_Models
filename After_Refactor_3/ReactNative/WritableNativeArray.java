/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.bridge;

import androidx.annotation.Nullable;
import com.facebook.infer.annotation.Assertions;
import com.facebook.jni.HybridData;
import com.facebook.proguard.annotations.DoNotStrip;

/**
 * This implementation of a write-only array stored in native memory inherits properties from
 * ReadableNativeArray. Use Arguments.createArray() to stub out the creation of this class for testing purposes.
 * (Note: #TODO 5815532 needs further investigation.)
 */
@DoNotStrip
public class WritableNativeArray extends ReadableNativeArray implements WritableArray {
  
  static {
    ReactBridge.staticInit();
  }

  /**
   * Initializes a new instance of the WritableNativeArray class.
   */
  public WritableNativeArray() {
    super(initHybrid());
  }

  /**
   * Adds a null value to the end of the array.
   */
  @Override
  public native void pushNull();

  /**
   * Adds a boolean value to the end of the array.
   * 
   * @param value The boolean value to add to the array.
   */
  @Override
  public native void pushBoolean(boolean value);

  /**
   * Adds a double value to the end of the array.
   * 
   * @param value The double value to add to the array.
   */
  @Override
  public native void pushDouble(double value);

  /**
   * Adds an integer value to the end of the array.
   * 
   * @param value The integer value to add to the array.
   */
  @Override
  public native void pushInt(int value);

  /**
   * Adds a string value to the end of the array.
   * 
   * @param value The string value to add to the array.
   */
  @Override
  public native void pushString(@Nullable String value);

  /**
   * Adds an array to the end of the array.
   * Note: This consumes the map so do not reuse it.
   * 
   * @param array The array to add to the end of the current array.
   */
  @Override
  public void pushArray(@Nullable ReadableArray array) {
    Assertions.assertCondition(
        array == null || array instanceof ReadableNativeArray, "Illegal type provided");
    pushNativeArray((ReadableNativeArray) array);
  }

  /**
   * Adds a map to the end of the array.
   * Note: This consumes the map so do not reuse it.
   * 
   * @param map The map to add to the end of the current array.
   */
  @Override
  public void pushMap(@Nullable ReadableMap map) {
    Assertions.assertCondition(
        map == null || map instanceof ReadableNativeMap, "Illegal type provided");
    pushNativeMap((ReadableNativeMap) map);
  }

  /**
   * Initializes the hybrid data for this array.
   * 
   * @return The hybrid data for this array.
   */
  private static native HybridData initHybrid();

  /**
   * Adds a native array to the end of the array.
   * Note: This consumes the array so do not reuse it.
   * 
   * @param array The native array to add to the end of the current array.
   */
  private native void pushNativeArray(ReadableNativeArray array);

  /**
   * Adds a native map to the end of the array.
   * Note: This consumes the map so do not reuse it.
   * 
   * @param map The native map to add to the end of the current array.
   */
  private native void pushNativeMap(ReadableNativeMap map);
  
} 