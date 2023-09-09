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
 * This class implements a write-only array that is stored in native memory. It provides methods to add
 * different types of values to the array. 
 */
@DoNotStrip
public class WritableNativeArray extends ReadableNativeArray implements WritableArray {

  /**
   * Initializes the class and allocates memory in native memory for the array.
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
   */
  @Override
  public native void pushBoolean(boolean value);

  /**
   * Adds a double value to the end of the array.
   */
  @Override
  public native void pushDouble(double value);

  /**
   * Adds an integer value to the end of the array.
   */
  @Override
  public native void pushInt(int value);

  /**
   * Adds a string value to the end of the array.
   */
  @Override
  public native void pushString(@Nullable String value);

  /**
   * Adds a readable array to the end of the array.
   *
   * @param array The readable array to add.
   * @throws RuntimeException if the array is not an instance of ReadableNativeArray or is null.
   */
  @Override
  public void pushArray(@Nullable ReadableArray array) {
    Assertions.assertCondition(
        array == null || array instanceof ReadableNativeArray, "Illegal type provided");
    pushNativeArray((ReadableNativeArray) array);
  }

  /**
   * Adds a readable map to the end of the array.
   *
   * @param map The readable map to add.
   * @throws RuntimeException if the map is not an instance of ReadableNativeMap or is null.
   */
  @Override
  public void pushMap(@Nullable ReadableMap map) {
    Assertions.assertCondition(
        map == null || map instanceof ReadableNativeMap, "Illegal type provided");
    pushNativeMap((ReadableNativeMap) map);
  }

  /**
   * Initializes the hybrid data needed to allocate memory in native memory for the array.
   */
  private static native HybridData initHybrid();

  /**
   * Adds a native array to the end of the array.
   *
   * @param array The native array to add.
   */
  private native void pushNativeArray(ReadableNativeArray array);

  /**
   * Adds a native map to the end of the array.
   *
   * @param map The native map to add.
   */
  private native void pushNativeMap(ReadableNativeMap map);
} 