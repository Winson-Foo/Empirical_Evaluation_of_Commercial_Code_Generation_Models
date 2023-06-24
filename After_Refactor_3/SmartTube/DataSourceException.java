package com.google.android.exoplayer2.upstream;

import java.io.IOException;

/**
 * Used to specify reason of a DataSource error.
 */
public final class DataSourceException extends IOException {

  public enum ErrorReason {
    POSITION_OUT_OF_RANGE
  }

  /**
   * The reason of this {@link DataSourceException}.
   */
  private final ErrorReason errorReason;

  /**
   * Constructs a DataSourceException with the given error reason.
   *
   * @param errorReason Reason of the error.
   * @throws IllegalArgumentException if an invalid error reason is provided.
   */
  public DataSourceException(ErrorReason errorReason) throws IllegalArgumentException {
    if (errorReason == null) {
      throw new IllegalArgumentException("Error reason cannot be null");
    }
    this.errorReason = errorReason;
  }

  /**
   * Returns the reason of this {@link DataSourceException}.
   */
  public ErrorReason getErrorReason() {
    return errorReason;
  }

}

