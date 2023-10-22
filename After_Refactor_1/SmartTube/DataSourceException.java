/*
 * Copyright (C) 2016 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.android.exoplayer2.upstream;

import java.io.IOException;

/**
 * An exception that specifies the reason of a DataSource error.
 */
public final class DataSourceException extends IOException {

  /**
   * The reason of this DataSourceException, indicating that the requested
   * position is out of the valid range of the data source.
   */
  public static final int POSITION_OUT_OF_RANGE = 0;

  /**
   * The reason of this DataSourceException.
   */
  public final int reason;

  /**
   * Constructs a new DataSourceException with the specified reason.
   *
   * @param reason The reason of the error, which can only be POSITION_OUT_OF_RANGE.
   */
  public DataSourceException(int reason) {
    this.reason = reason;
  }

}