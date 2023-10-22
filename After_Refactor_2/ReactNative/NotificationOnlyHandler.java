/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.packagerconnection;

import com.facebook.common.logging.FLog;

/**
 * Handles notifications without handling requests.
 */
public abstract class NotificationHandler implements RequestHandler {
  private static final String TAG = NotificationHandler.class.getSimpleName();

  /**
   * Handles a request by sending an error and logging an error message.
   *
   * @param params Unused.
   * @param responder The responder to use to send the error.
   */
  public final void handleRequest(Object params, Responder responder) {
    responder.error("Requests are not supported");
    FLog.e(TAG, "Request is not supported");
  }

  /**
   * Handles a notification.
   *
   * @param params The parameters of the notification.
   */
  public abstract void handleNotification(Object params);
}