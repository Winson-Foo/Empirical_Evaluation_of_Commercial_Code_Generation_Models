package com.facebook.react.uimanager;

import android.view.View;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import com.facebook.react.bridge.JSApplicationCausedNativeException;

/**
 * Exception thrown when the view operation requested by JS is illegal.
 */
public class IllegalViewOperationException extends JSApplicationCausedNativeException {

  @Nullable private final View view;

  public IllegalViewOperationException(@NonNull String message) {
    super(message);
    this.view = null;
  }

  public IllegalViewOperationException(@NonNull String message, @Nullable View view, @Nullable Throwable cause) {
    super(message, cause);
    this.view = view;
  }

  /**
   * Gets the view where the illegal operation was requested.
   *
   * @return The view where the illegal operation was requested, or null if not available.
   */
  @Nullable
  public View getView() {
    return view;
  }
}