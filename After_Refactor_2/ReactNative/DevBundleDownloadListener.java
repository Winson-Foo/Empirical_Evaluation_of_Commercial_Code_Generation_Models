// Refactored code:

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.devsupport.interfaces;

import androidx.annotation.Nullable;

public interface DevBundleDownloadListener {
  /**
   * This method is called when the bundle download is successful.
   */
  void onSuccess();

  /**
   * This method is called to update the progress of the bundle download.
   * 
   * @param status - status message to display
   * @param done   - number of bytes downloaded
   * @param total  - total number of bytes to download
   */
  void onProgress(@Nullable String status, @Nullable Integer done, @Nullable Integer total);

  /**
   * This method is called when an error occurs during the bundle download.
   * 
   * @param cause - exception that caused the download failure
   */
  void onFailure(Exception cause);
}

