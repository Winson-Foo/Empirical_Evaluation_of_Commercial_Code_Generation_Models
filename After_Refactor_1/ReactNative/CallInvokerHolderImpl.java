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
* This class is responsible for holding the JSCallInvoker.
*/
public class CallInvokerHolderImpl implements CallInvokerHolder {
  
  // Prevents multiple simultaneous initialization of the SoLoader library.
  private static volatile boolean sIsSoLibraryLoaded;

  private final HybridData mHybridData;

  /**
  * Creates a new instance of CallInvokerHolderImpl with the given HybridData.
  */
  private CallInvokerHolderImpl(HybridData hybridData) {
    maybeLoadSoLibrary();
    mHybridData = hybridData;
  }

  /**
  * Loads the SoLoader library if it has not already been loaded.
  */
  private static synchronized void maybeLoadSoLibrary() {
    if (!sIsSoLibraryLoaded) {
      SoLoader.loadLibrary("turbomodulejsijni");
      sIsSoLibraryLoaded = true;
    }
  }
}

public class CallInvokerHolderImpl implements CallInvokerHolder {

  private final HybridData mHybridData;

  public CallInvokerHolderImpl(HybridData hybridData, SoLoaderWrapper soLoader) {
    soLoader.loadLibraryIfNeeded();
    mHybridData = hybridData;
  }
}

public class SoLoaderWrapper {
  private volatile boolean isSoLibraryLoaded;

  public synchronized void loadLibraryIfNeeded() {
    if (!isSoLibraryLoaded) {
      SoLoader.loadLibrary("turbomodulejsijni");
      isSoLibraryLoaded = true;
    }
  }
}