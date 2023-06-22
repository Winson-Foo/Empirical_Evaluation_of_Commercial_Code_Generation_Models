package com.facebook.react.uimanager.events;

import android.view.MotionEvent;
import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.ReactSoftExceptionLogger;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.uimanager.PixelUtil;
import com.facebook.systrace.Systrace;

public class TouchesHelper {

  private static final String TAG = "TouchesHelper";

  private static final String PAGE_X_KEY = "pageX";
  private static final String PAGE_Y_KEY = "pageY";
  private static final String TIMESTAMP_KEY = "timestamp";
  private static final String POINTER_IDENTIFIER_KEY = "identifier";
  private static final String LOCATION_X_KEY = "locationX";
  private static final String LOCATION_Y_KEY = "locationY";
  
  private static final String TARGET_SURFACE_KEY = "targetSurface";
  private static final String TARGET_KEY = "target";
  private static final String CHANGED_TOUCHES_KEY = "changedTouches";
  private static final String TOUCHES_KEY = "touches";

  /**
   * Creates catalyst pointers array in format that is expected by RCTEventEmitter JS module from
   * given {@param event} instance. This method use {@param reactTarget} parameter to set as a
   * target view id associated with current gesture.
   */
  private static WritableMap[] createPointersArray(TouchEvent event) {
    MotionEvent motionEvent = event.getMotionEvent();
    WritableMap[] pointers = new WritableMap[motionEvent.getPointerCount()];

    float targetX = motionEvent.getX() - event.getViewX();
    float targetY = motionEvent.getY() - event.getViewY();

    for (int index = 0; index < motionEvent.getPointerCount(); index++) {
      WritableMap pointer = Arguments.createMap();

      pointer.putDouble(PAGE_X_KEY, PixelUtil.toDIPFromPixel(motionEvent.getX(index)));
      pointer.putDouble(PAGE_Y_KEY, PixelUtil.toDIPFromPixel(motionEvent.getY(index)));

      float locationX = motionEvent.getX(index) - targetX;
      float locationY = motionEvent.getY(index) - targetY;
      pointer.putDouble(LOCATION_X_KEY, PixelUtil.toDIPFromPixel(locationX));
      pointer.putDouble(LOCATION_Y_KEY, PixelUtil.toDIPFromPixel(locationY));

      pointer.putInt(TARGET_SURFACE_KEY, event.getSurfaceId());
      pointer.putInt(TARGET_KEY, event.getViewTag());
      pointer.putDouble(TIMESTAMP_KEY, event.getTimestampMs());
      pointer.putDouble(POINTER_IDENTIFIER_KEY, motionEvent.getPointerId(index));

      pointers[index] = pointer;
    }

    return pointers;
  }

  /**
   * Generate and send touch event to RCTEventEmitter JS module associated with the given {@param
   * context} for legacy renderer. Touch event can encode multiple concurrent pointers.
   *
   * @param rctEventEmitter Event emitter used to execute JS module call
   * @param touchEvent native touch event to read pointers count and coordinates from
   */
  public static void sendTouchesLegacy(RCTEventEmitter rctEventEmitter, TouchEvent touchEvent) {
    TouchEventType type = touchEvent.getTouchEventType();

    WritableArray pointers = getWritableArray(/* copyObjects */ false, createPointersArray(touchEvent));
    MotionEvent motionEvent = touchEvent.getMotionEvent();

    WritableArray changedIndices = Arguments.createArray();
    if (type == TouchEventType.MOVE || type == TouchEventType.CANCEL) {
      for (int i = 0; i < motionEvent.getPointerCount(); i++) {
        changedIndices.pushInt(i);
      }
    } else if (type == TouchEventType.START || type == TouchEventType.END) {
      changedIndices.pushInt(motionEvent.getActionIndex());
    } else {
      throw new RuntimeException("Unknown touch type: " + type);
    }

    rctEventEmitter.receiveTouches(TouchEventType.getJSEventName(type), pointers, changedIndices);
  }

  /**
   * Generate touch event data to match JS expectations. Combines logic in {@link #sendTouchEvent}
   * and FabricEventEmitter to create the same data structure in a more efficient manner.
   *
   * <p>Touches have to be dispatched as separate events for each changed pointer to make JS process
   * them correctly. To avoid allocations, we preprocess touch events in Java world and then convert
   * them to native before dispatch.
   *
   * @param eventEmitter emitter to dispatch event to
   * @param event the touch event to extract data from
   */
  public static void sendTouchEvent(RCTModernEventEmitter eventEmitter, TouchEvent event) {
    Systrace.beginSection(Systrace.TRACE_TAG_REACT_JAVA_BRIDGE, 
        "TouchesHelper.sentTouchEventModern(" + event.getEventName() + ")");
    try {
      TouchEventType type = event.getTouchEventType();
      MotionEvent motionEvent = event.getMotionEvent();
      if (motionEvent == null) {
        ReactSoftExceptionLogger.logSoftException(TAG,
            new IllegalStateException("Cannot dispatch a TouchEvent that has no MotionEvent; the TouchEvent has been recycled"));
        return;
      }

      WritableMap[] pointers = createPointersArray(event);
      WritableMap[] changedPointers = null;

      switch (type) {
        case START:
          int newPointerIndex = motionEvent.getActionIndex();
          changedPointers = new WritableMap[] {pointers[newPointerIndex].copy()};
          break;

        case END:
          int finishedPointerIndex = motionEvent.getActionIndex();
          WritableMap finishedPointer = pointers[finishedPointerIndex];
          pointers[finishedPointerIndex] = null;
          changedPointers = new WritableMap[] {finishedPointer};
          break;

        case MOVE:
          changedPointers = new WritableMap[pointers.length];
          for (int i = 0; i < pointers.length; i++) {
            changedPointers[i] = pointers[i].copy();
          }
          break;

        case CANCEL:
          changedPointers = pointers;
          pointers = new WritableMap[0];
          break;
      }

      for (WritableMap pointerData : changedPointers) {
        WritableMap eventData = pointerData.copy();
        WritableArray changedPointersArray = getWritableArray(/* copyObjects */ true, changedPointers);
        WritableArray pointersArray = getWritableArray(/* copyObjects */ true, pointers);

        eventData.putArray(CHANGED_TOUCHES_KEY, changedPointersArray);
        eventData.putArray(TOUCHES_KEY, pointersArray);

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

  private static WritableArray getWritableArray(boolean copyObjects, WritableMap... objects) {
    WritableArray result = Arguments.createArray();
    for (WritableMap object : objects) {
      if (object != null) {
        result.pushMap(copyObjects ? object.copy() : object);
      }
    }
    return result;
  }
} 