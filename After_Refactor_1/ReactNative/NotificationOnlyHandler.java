/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.packagerconnection;

public abstract class NotificationHandler implements RequestHandler {

  public final void onRequest(Object params, Responder responder) {
    responder.error("Request is not supported");
  }

  public abstract void onNotification(Object params);
}