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
   * Called when the dev bundle download is successful.
   */
  public void onSuccess();

  /**
   * Called to notify progress of the dev bundle download.
   * @param status The status of the download.
   * @param done The amount of the download that is completed.
   * @param total The total amount of the download.
   */
  public void onProgress(@Nullable String status, @Nullable Integer done, @Nullable Integer total);

  /**
   * Called when the dev bundle download fails.
   * @param cause The exception that caused the failure.
   */
  public void onFailure(Exception cause);
}

