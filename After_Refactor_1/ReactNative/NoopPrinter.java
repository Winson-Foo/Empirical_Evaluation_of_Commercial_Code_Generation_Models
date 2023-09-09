/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.debug.holder;

import com.facebook.debug.debugoverlay.model.DebugOverlayTag;

/**
 * No-op implementation of {@link Printer}.
 */
public class NoopPrinter implements Printer {

  public static final NoopPrinter INSTANCE = new NoopPrinter();

  public NoopPrinter() {
    // Constructor made public
  }

  @Override
  public void logMessage(final DebugOverlayTag tag, final String message, final Object... args) {
    // No-op
  }

  @Override
  public void logMessage(final DebugOverlayTag tag, final String message) {
    // No-op
  }

  @Override
  public boolean shouldDisplayLogMessage(final DebugOverlayTag tag) {
    return false;
  }
}