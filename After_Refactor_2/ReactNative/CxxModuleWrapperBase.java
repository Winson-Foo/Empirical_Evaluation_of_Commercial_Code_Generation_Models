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
 * A wrapper for a cross-platform C++ module.
 * <p>
 * This module implements the NativeModule interface but will never be invoked from Java,
 * instead the underlying Cxx module will be extracted by the bridge and called directly.
 */
@DoNotStrip
public class CxxModuleWrapperBase implements NativeModule {
    
    @DoNotStrip 
    private HybridData mHybridData;
    
    static {
        ReactBridge.staticInit();
    }

    /**
     * Gets the name of the module.
     * 
     * @return The name of the module.
     */
    @Override
    public native String getName();

    /**
     * Initializes the module.
     */
    @Override
    public void initialize() {
        // do nothing
    }

    /**
     * Determines if this module can override an existing one.
     * 
     * @return True if this module can override an existing one, false otherwise.
     */
    @Override
    public boolean canOverrideExistingModule() {
        return false;
    }

    /**
     * Called when the CatalystInstance is destroyed.
     */
    @Override
    public void onCatalystInstanceDestroy() {}

    /**
     * Invalidates the module.
     */
    @Override
    public void invalidate() {
        mHybridData.resetNative();
    }

    /**
     * Constructs a CxxModuleWrapperBase instance.
     * 
     * @param hd The HybridData to use.
     */
    protected CxxModuleWrapperBase(HybridData hd) {
        mHybridData = hd;
    }

    /**
     * Replaces the current native module held by this wrapper by a new instance.
     * 
     * @param hd The new HybridData instance.
     */
    protected void resetModule(HybridData hd) {
        if (hd != mHybridData) {
            mHybridData.resetNative();
            mHybridData = hd;
        }
    }
}

