/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.uimanager;

import android.view.ViewGroup;

import com.facebook.react.viewmanagers.InterfaceOnlyNativeComponentViewManagerDelegate;
import com.facebook.react.viewmanagers.InterfaceOnlyNativeComponentViewManagerInterface;

public class InterfaceOnlyNativeComponentViewManager 
        extends SimpleViewManager<ViewGroup> 
        implements InterfaceOnlyNativeComponentViewManagerInterface<ViewGroup> {

    public static final String REACT_CLASS = "InterfaceOnlyNativeComponentView";

    // Returns the name of the native component 
    @Override
    public String getName() {
        return REACT_CLASS;
    }

    // Creates a new instance of the native component 
    @Override
    public ViewGroup createViewInstance(ThemedReactContext context) {
        // No instance created 
        return null;
    }

    // Sets the title of the native component 
    @Override
    public void setTitle(ViewGroup view, String title) {
        // Set the title
    }

}