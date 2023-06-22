/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.views.textinput;

/**
 * An interface that defines a method for monitoring changes in the size of the content in a text input view.
 */
public interface TextInputContentSizeWatcher {
  
  /**
   * Called when the layout of the content in the text input view has changed.
   */
  void onLayout();
  
}

