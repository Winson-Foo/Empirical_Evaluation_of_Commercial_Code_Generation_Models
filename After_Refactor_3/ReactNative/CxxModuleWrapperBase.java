/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.bridge;

import com.facebook.jni.HybridData;
import com.facebook.proguard.annotations.DoNotStrip;

/**
 * A Java Object which represents a cross-platform C++ module.
 *
 * <p>This module implements the NativeModule interface but will never be invoked from Java, instead
 * the underlying Cxx module will be extracted by the bridge and called directly.
 */
@DoNotStrip
public class CxxModuleWrapperBase implements NativeModule {

  private final HybridData mHybridData;

  /**
   * Construct a new instance of CxxModuleWrapperBase.
   */
  public CxxModuleWrapperBase(HybridData hd) {
    mHybridData = hd;
    ReactBridge.staticInit();
  }

  /**
   * Get the name of this module.
   */
  @Override
  public native String getName();

  /**
   * Initialize the module. This method does nothing.
   */
  @Override
  public void initialize() {
    // Do nothing.
  }

  /**
   * Check whether this module can override an existing module.
   */
  @Override
  public boolean canOverrideExistingModule() {
    return false;
  }

  /**
   * Clean up the module.
   */
  @Override
  public void onCatalystInstanceDestroy() {
    resetModule(null);
  }

  /**
   * Reset the underlying native module.
   */
  public synchronized void resetModule(HybridData hd) {
    if (mHybridData != null) {
      invalidate();
    }
    if (hd != null) {
      mHybridData.resetNative();
      mHybridData.assign(hd);
    }
  }

  private synchronized void invalidate() {
    if (mHybridData != null) {
      mHybridData.resetNative();
    }
  }

}