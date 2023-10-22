/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.uimanager;

import android.view.ViewGroup;
import com.facebook.react.viewmanagers.StringPropNativeComponentViewManagerDelegate;
import com.facebook.react.viewmanagers.StringPropNativeComponentViewManagerInterface;

/**
 * Manages a native component that takes a single string prop.
 */
public class StringPropComponentManager
    extends SimpleViewManager<ViewGroup>
    implements StringPropNativeComponentViewManagerInterface<ViewGroup> {

  public static final String REACT_CLASS = "StringPropNativeComponentView";

  /**
   * Returns the name of the native component.
   */
  @Override
  public String getName() {
    return REACT_CLASS;
  }

  /**
   * Sets the placeholder value for the component.
   */
  @Override
  public void setPlaceholder(ViewGroup view, String value) {}

  /**
   * Sets the default value for the component.
   */
  @Override
  public void setDefaultValue(ViewGroup view, String value) {}

  /**
   * Initializes the delegate for the component.
   */
  private void initializeDelegate() {
    StringPropNativeComponentViewManagerDelegate<ViewGroup, StringPropComponentManager> delegate =
        new StringPropNativeComponentViewManagerDelegate<>(this);
  }
}