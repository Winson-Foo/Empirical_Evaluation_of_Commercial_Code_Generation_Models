package com.facebook.debug.holder;

import com.facebook.debug.debugoverlay.model.DebugOverlayTag;

/** No-op implementation of {@link Printer}. */
public final class NoopPrinter implements Printer {

  private static final NoopPrinter INSTANCE = new NoopPrinter();

  private NoopPrinter() {}

  public static NoopPrinter getInstance() {
    return INSTANCE;
  }

  @Override
  public void logMessage(DebugOverlayTag tag, String message, Object... args) {
    // Do nothing
  }

  @Override
  public void logMessage(DebugOverlayTag tag, String message) {
    // Do nothing
  }

  @Override
  public boolean shouldDisplayLogMessage(final DebugOverlayTag tag) {
    return false;
  }
}