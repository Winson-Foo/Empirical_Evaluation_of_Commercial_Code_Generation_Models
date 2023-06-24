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

/**
 * This view manager represents a simple interface-only native component, 
 * that can be used to render native views with no React components inside.
 */
public class InterfaceOnlyNativeComponentViewManager extends SimpleViewManager<ViewGroup>
    implements InterfaceOnlyNativeComponentViewManagerInterface<ViewGroup> {

  public static final String INTERFACE_VIEW_CLASS = "InterfaceOnlyNativeComponentView";

  @Override
  public String getName() {
    return INTERFACE_VIEW_CLASS;
  }

  /**
   * Creates a delegate for this view manager.
   */
  private void createDelegate() {
    Delegate<ViewGroup, InterfaceOnlyNativeComponentViewManager> delegate =
        new Delegate<>(this);
  }

  @Override
  public ViewGroup createViewInstance(ThemedReactContext context) {
    throw new IllegalStateException();
  }

  /**
   * Sets the title of the interface-only native component.
   */
  @Override
  public void setInterfaceTitle(ViewGroup view, String value) {}
} 