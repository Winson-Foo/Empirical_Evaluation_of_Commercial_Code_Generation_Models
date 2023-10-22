/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.turbomodule.core;

import com.facebook.jni.HybridData;
import com.facebook.react.turbomodule.core.interfaces.CallInvokerHolder;
import com.facebook.soloader.SoLoader;

/**
 * Wrapper class for JSCallInvoker that can be passed from CatalystInstance to TurboModuleManager.
 */
public class CallInvokerHolderImpl implements CallInvokerHolder {

  // Flag to check if SO library is already loaded
  private static volatile boolean isSoLibraryLoaded;

  // JNI hybrid data
  private final HybridData hybridData;

  /**
   * Constructs a new CallInvokerHolderImpl object.
   *
   * @param hd The JNI hybrid data.
   */
  private CallInvokerHolderImpl(HybridData hd) {
    loadSoLibraryIfNeeded();
    hybridData = hd;
  }

  /**
   * Loads the SO library if it's not already loaded.
   */
  private static synchronized void loadSoLibraryIfNeeded() {
    if (!isSoLibraryLoaded) {
      try {
        SoLoader.loadLibrary("turbomodulejsijni");
        isSoLibraryLoaded = true;
      } catch (UnsatisfiedLinkError e) {
        // TODO: Handle the error
      }
    }
  }
}

