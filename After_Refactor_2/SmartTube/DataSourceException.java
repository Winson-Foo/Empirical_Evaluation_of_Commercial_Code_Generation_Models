package com.google.android.exoplayer2.upstream;

import androidx.annotation.IntDef;

import java.io.IOException;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

/**
 * Used to specify reason of a DataSource error.
 */
public final class DataSourceException extends IOException {

  @Retention(RetentionPolicy.SOURCE)
  @IntDef({ERROR_POSITION_OUT_OF_RANGE})
  public @interface Reason {}

  public static final int ERROR_POSITION_OUT_OF_RANGE = 0;

  /**
   * The reason of this {@link DataSourceException}.
   */
  @Reason
  public final int reason;

  /**
   * Constructs a DataSourceException.
   *
   * @param reason Reason of the error. It can only be {@link #ERROR_POSITION_OUT_OF_RANGE}.
   */
  public DataSourceException(@Reason int reason) {
    this.reason = reason;
  }

} 