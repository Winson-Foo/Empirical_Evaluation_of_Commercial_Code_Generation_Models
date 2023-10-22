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

public class StringPropNativeComponentViewManager extends SimpleViewManager<ViewGroup>
    implements ViewManagerInterface<ViewGroup> {

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

interface StringPropNativeComponentViewManagerInterface<T extends ViewGroup> extends ViewManagerInterface<T> {
  void setPlaceholder(T view, String value);
  void setDefaultValue(T view, String value);
} 