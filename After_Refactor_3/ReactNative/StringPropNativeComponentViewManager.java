/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.uimanager;

import android.view.ViewGroup;
import com.facebook.react.viewmanagers.StringPropNativeComponentViewManagerInterface;

public class StringPropNativeComponentViewManager extends SimpleViewManager<ViewGroup>
    implements StringPropNativeComponentViewManagerInterface<ViewGroup> {

  public static final String REACT_CLASS = "StringPropNativeComponentView";

  @Override
  public String getName() {
    return REACT_CLASS;
  }

  @Override
  public ViewGroup createViewInstance(ThemedReactContext context) {
    throw new IllegalStateException();
  }

  @Override
  public void setPlaceholder(ViewGroup view, String value) {}

  @Override
  public void setDefaultValue(ViewGroup view, String value) {}
} 

package com.facebook.react.viewmanagers;

import android.view.ViewGroup;

public class StringPropNativeComponentViewManagerDelegate<V extends ViewGroup, T extends StringPropNativeComponentViewManagerInterface<V>> {

  private final T mViewManager;

  public StringPropNativeComponentViewManagerDelegate(T viewManager) {
      mViewManager = viewManager;
  }

  public void setPlaceholder(V view, String value) {
    mViewManager.setPlaceholder(view, value);
  }

  public void setDefaultValue(V view, String value) {
    mViewManager.setDefaultValue(view, value);
  }

} 