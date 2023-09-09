/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.uimanager;

import android.view.View;
import androidx.annotation.Nullable;
import com.facebook.react.bridge.JSApplicationCausedNativeException;

/**
 * An exception caused by JS requesting the UI manager to perform an illegal view operation.
 */
public class IllegalViewOperationException extends JSApplicationCausedNativeException {

  /**
   * The view that caused the exception.
   */
  @Nullable
  private final View view;

  /**
   * Constructs a new IllegalViewOperationException with the specified message.
   *
   * @param message the error message.
   */
  public IllegalViewOperationException(String message) {
    super(message);
    view = null;
  }

  /**
   * Constructs a new IllegalViewOperationException with the specified message, view and cause.
   *
   * @param message the error message.
   * @param view the view that caused the exception.
   * @param cause the cause of the exception.
   */
  public IllegalViewOperationException(String message, @Nullable View view, Throwable cause) {
    super(message, cause);
    this.view = view;
  }

  /**
   * Returns the view that caused the exception.
   *
   * @return the view that caused the exception or null if view was not specified.
   */
  @Nullable
  public View getView() {
    return view;
  }
}

