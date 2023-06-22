// CallInvokerHolderImpl.java
package com.facebook.react.turbomodule.core;

import com.facebook.jni.HybridData;
import com.facebook.react.turbomodule.core.interfaces.CallInvokerHolder;
import com.facebook.react.turbomodule.core.utils.SoLoaderUtils;

public class CallInvokerHolderImpl implements CallInvokerHolder {
  private final HybridData mHybridData;

  public CallInvokerHolderImpl(HybridData hd) {
    SoLoaderUtils.maybeLoadLibrary("turbomodulejsijni");
    mHybridData = hd;
  }

  public HybridData getHybridData() {
    return mHybridData;
  }
}

This makes the codebase more modular, testable, and easier to maintain in the long run.

