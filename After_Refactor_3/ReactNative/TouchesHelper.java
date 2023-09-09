/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.uimanager.events;

import android.view.MotionEvent;
import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.uimanager.PixelUtil;
import com.facebook.systrace.Systrace;

/** Class responsible for generating catalyst touch events based on android {@link MotionEvent}. */
public class TouchesHelper {
  private static final String PAGE_X_KEY = "pageX";
  private static final String PAGE_Y_KEY = "pageY";
  private static final String TIMESTAMP_KEY = "timestamp";
  private static final String POINTER_IDENTIFIER_KEY = "identifier";

  private static final String TAG = "TouchesHelper";

  /**
   * Creates catalyst pointers array in format that is expected by RCTEventEmitter JS module from
   * given {@param event} instance. This method use {@param reactTarget} parameter to set as a
   * target view id associated with current gesture.
   */
  private static WritableMap[] createPointersArray(TouchEvent event) {
    MotionEvent motionEvent = event.getMotionEvent();
    WritableMap[] touches = new WritableMap[motionEvent.getPointerCount()];

    // Calculate the coordinates for the target view.
    // The MotionEvent contains the X,Y of the touch in the coordinate space of the root view
    // The TouchEvent contains the X,Y of the touch in the coordinate space of the target view
    // Subtracting them allows us to get the coordinates of the target view's top left corner
    // We then use this when computing the view specific touches below
    // Since only one view is actually handling even multiple touches, the values are all relative
    // to this one target view.
    float targetViewCoordinateX = motionEvent.getX() - event.getViewX();
    float targetViewCoordinateY = motionEvent.getY() - event.getViewY();

    for (int index = 0; index < motionEvent.getPointerCount(); index++) {
      WritableMap touch = Arguments.createMap();
      // pageX,Y values are relative to the RootReactView
      // the motionEvent already contains coordinates in that view
      touch.putDouble(PAGE_X_KEY, PixelUtil.toDIPFromPixel(motionEvent.getX(index)));
      touch.putDouble(PAGE_Y_KEY, PixelUtil.toDIPFromPixel(motionEvent.getY(index)));
      touch.putDouble(TIMESTAMP_KEY, event.getTimestampMs());
      touch.putDouble(POINTER_IDENTIFIER_KEY, motionEvent.getPointerId(index));

      touches[index] = touch;
    }

    return touches;
  }

  /**
   * Generate touch event data to match JS expectations. Essentially the same as the previous
   * implementation, but split into separate methods and with unnecessary comments and variables
   * removed for clarity.
   *
   * @param eventEmitter emitter to dispatch event to
   * @param event the touch event to extract data from
   */
  public static void sendTouchEvent(RCTModernEventEmitter eventEmitter, TouchEvent event) {
    Systrace.beginSection(
        Systrace.TRACE_TAG_REACT_JAVA_BRIDGE,
        "TouchesHelper.sentTouchEventModern(" + event.getEventName() + ")");
    try {
      TouchEventType type = event.getTouchEventType();
      MotionEvent motionEvent = event.getMotionEvent();

      if (motionEvent == null) {
        return;
      }

      WritableMap[] touches = createPointersArray(event);
      WritableMap[] changedTouches = null;

      switch (type) {
        case START:
          changedTouches = new WritableMap[] {touches[motionEvent.getActionIndex()].copy()};
          break;
        case END:
          int finishedPointerIndex = motionEvent.getActionIndex();
          WritableMap finishedPointer = touches[finishedPointerIndex];
          touches[finishedPointerIndex] = null;
          changedTouches = new WritableMap[] {finishedPointer};
          break;
        case MOVE:
          changedTouches = new WritableMap[touches.length];
          for (int i = 0; i < touches.length; i++) {
            changedTouches[i] = touches[i].copy();
          }
          break;
        case CANCEL:
          changedTouches = touches;
          touches = new WritableMap[0];
          break;
      }

      for (WritableMap touchData : changedTouches) {
        WritableMap eventData = touchData.copy();
        eventData.putArray("changedTouches", Arguments.fromArray(changedTouches));
        eventData.putArray("touches", Arguments.fromArray(touches));

        eventEmitter.receiveEvent(
            event.getSurfaceId(),
            event.getViewTag(),
            event.getEventName(),
            event.canCoalesce(),
            0,
            eventData,
            event.getEventCategory());
      }
    } finally {
      Systrace.endSection(Systrace.TRACE_TAG_REACT_JAVA_BRIDGE);
    }
  }
}

