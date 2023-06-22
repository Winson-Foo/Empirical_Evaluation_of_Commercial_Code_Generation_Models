/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.modules.core;

import android.app.Activity;

/**
 * Interface used by activities to delegate permission request results.
 * Classes implementing this interface will be notified whenever there's a result for a permission request.
 */
public interface PermissionListener {
  
  /**
   * Method called whenever there's a result to a permission request.
   * It is forwarded from the onRequestPermissionsResult method of the Activity class.
   *
   * @param requestCode The request code passed in the requestPermissions method.
   * @param permissions The requested permissions.
   * @param grantResults The grant results for the corresponding permissions.
   * @return True if the PermissionListener can be removed, false otherwise.
   */
  boolean onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults);
}

