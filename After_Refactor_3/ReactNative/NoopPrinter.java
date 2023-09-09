package com.facebook.debug.holder;

import com.facebook.debug.debugoverlay.model.DebugOverlayTag;

/** 
 * A printer that does nothing. 
 */
public final class NoOpPrinter implements Printer {

  /** Singleton instance. */
  private static final NoOpPrinter INSTANCE = new NoOpPrinter();

  private NoOpPrinter() {}

  /**
   * Returns the singleton instance.
   */
  public static NoOpPrinter getInstance() {
    return INSTANCE;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public void logMessage(DebugOverlayTag tag, String message, Object... args) {
    // do nothing
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public void logMessage(DebugOverlayTag tag, String message) {
    // do nothing
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public boolean shouldDisplayLogMessage(final DebugOverlayTag tag) {
    return false;
  }
}

