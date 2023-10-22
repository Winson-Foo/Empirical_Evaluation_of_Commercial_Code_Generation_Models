/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.packagerconnection;

import com.facebook.common.logging.FLog;

public abstract class NotificationHandler implements RequestHandler {
  private static final String TAG = NotificationHandler.class.getSimpleName();

  @Override
  public void onRequest(Object params, Responder responder) {
    FLog.e(TAG, "Request is not supported");
  }

  public abstract void onNotification(Object params);
} 