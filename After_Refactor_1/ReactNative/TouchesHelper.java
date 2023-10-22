package com.facebook.react.uimanager.events;

import android.view.MotionEvent;
import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.ReactSoftExceptionLogger;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.uimanager.PixelUtil;
import com.facebook.systrace.Systrace;

/**
 * Class responsible for generating catalyst touch events based on android {@link MotionEvent}.
 */
public class TouchesHelper {
  // Constants for magic strings and numbers.
  private static final String TAG = "TouchesHelper";
  private static final String TARGET_SURFACE_KEY = "targetSurface";
  private static final String TARGET_KEY = "target";
  private static final String CHANGED_TOUCHES_KEY = "changedTouches";
  private static final String TOUCHES_KEY = "touches";
  private static final String PAGE_X_KEY = "pageX";
  private static final String PAGE_Y_KEY = "pageY";
  private static final String TIMESTAMP_KEY = "timestamp";
  private static final String POINTER_IDENTIFIER_KEY = "identifier";
  private static final String LOCATION_X_KEY = "locationX";
  private static final String LOCATION_Y_KEY = "locationY";

  /**
   * Creates catalyst pointers array in format that is expected by RCTEventEmitter JS module from
   * given {@param event} instance. This method use {@param reactTarget} parameter to set as a
   * target view id associated with current gesture.
   */
  private static WritableMap[] createPointersArray(TouchEvent event) {
    MotionEvent motionEvent = event.getMotionEvent();
    WritableMap[] touches = new WritableMap[motionEvent.getPointerCount()];

    // Calculate the coordinates for the target view based on the MotionEvent X,Y and the TargetView X,Y.
    float targetViewCoordinateX = motionEvent.getX() - event.getViewX();
    float targetViewCoordinateY = motionEvent.getY() - event.getViewY();
    
    // Create a WritableMap for each pointer in the MotionEvent.
    for (int index = 0; index < motionEvent.getPointerCount(); index++) {
      WritableMap touch = Arguments.createMap();
      
      // Set pageX and pageY to the coordinates of the touch in the RootReactView.
      touch.putDouble(PAGE_X_KEY, PixelUtil.toDIPFromPixel(motionEvent.getX(index)));
      touch.putDouble(PAGE_Y_KEY, PixelUtil.toDIPFromPixel(motionEvent.getY(index)));
      
      // Set locationX and locationY to the coordinates of the touch in the target view.
      float locationX = motionEvent.getX(index) - targetViewCoordinateX;
      float locationY = motionEvent.getY(index) - targetViewCoordinateY;
      touch.putDouble(LOCATION_X_KEY, PixelUtil.toDIPFromPixel(locationX));
      touch.putDouble(LOCATION_Y_KEY, PixelUtil.toDIPFromPixel(locationY));
      
      // Set the surface id, target id, timestamp, and pointer identifier.
      touch.putInt(TARGET_SURFACE_KEY, event.getSurfaceId());
      touch.putInt(TARGET_KEY, event.getViewTag());
      touch.putDouble(TIMESTAMP_KEY, event.getTimestampMs());
      touch.putDouble(POINTER_IDENTIFIER_KEY, motionEvent.getPointerId(index));

      touches[index] = touch;
    }
    return touches;
  }

  /**
   * Generate and send touch event to RCTEventEmitter JS module associated with the given {@param
   * context} for legacy renderer. Touch event can encode multiple concurrent touches (pointers).
   *
   * @param rctEventEmitter Event emitter used to execute JS module call
   * @param touchEvent native touch event to read pointers count and coordinates from
   */
  public static void sendTouchesLegacy(RCTEventEmitter rctEventEmitter, TouchEvent touchEvent) {
    // Get the TouchEventType from the touchEvent.
    TouchEventType type = touchEvent.getTouchEventType();
    MotionEvent motionEvent = touchEvent.getMotionEvent();

    // Create the pointers array.
    WritableMap[] pointersArray = createPointersArray(touchEvent);

    // For START and END events, send only the index of the pointer that is associated with that event.
    // For MOVE and CANCEL events, 'changedIndices' array should contain all the pointers indices.
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

    // Emit the event using the event emitter.
    rctEventEmitter.receiveTouches(
      TouchEventType.getJSEventName(type), 
      getWritableArray(false, pointersArray),
      changedIndices
    );
  }

  /**
   * Generate touch event data to match JS expectations.
   *
   * @param eventEmitter emitter to dispatch event to
   * @param event the touch event to extract data from
   */
  public static void sendTouchEvent(RCTModernEventEmitter eventEmitter, TouchEvent event) {
    // Start the Systrace section.
    Systrace.beginSection(
        Systrace.TRACE_TAG_REACT_JAVA_BRIDGE,
        "TouchesHelper.sentTouchEventModern(" + event.getEventName() + ")");
    try {
      // Get the TouchEvent type and the MotionEvent.
      TouchEventType type = event.getTouchEventType();
      MotionEvent motionEvent = event.getMotionEvent();

      // Return if there is no MotionEvent.
      if (motionEvent == null) {
        ReactSoftExceptionLogger.logSoftException(
            TAG,
            new IllegalStateException(
                "Cannot dispatch a TouchEvent that has no MotionEvent; the TouchEvent has been recycled"));
        return;
      }

      // Create the touches and changedTouches arrays.
      WritableMap[] touches = createPointersArray(event);
      WritableMap[] changedTouches = null;

      switch (type) {
        case START:
          int newPointerIndex = motionEvent.getActionIndex();

          changedTouches = new WritableMap[] {touches[newPointerIndex].copy()};
          break;
        case END:
          int finishedPointerIndex = motionEvent.getActionIndex();
          /*
           * Clear finished pointer index for compatibility with W3C touch "end" events, where the
           * active touches don't include the set that has just been "ended".
           */
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

      // Send the event using the event emitter.
      for (WritableMap touchData : changedTouches) {
        WritableMap eventData = touchData.copy();
        WritableArray changedTouchesArray = getWritableArray(true, changedTouches);
        WritableArray touchesArray = getWritableArray(true, touches);

        eventData.putArray(CHANGED_TOUCHES_KEY, changedTouchesArray);
        eventData.putArray(TOUCHES_KEY, touchesArray);

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
      // End the Systrace section.
      Systrace.endSection(Systrace.TRACE_TAG_REACT_JAVA_BRIDGE);
    }
  }

  /**
   * Returns a WritableArray containing the objects in the input array.
   *
   * @param copyObjects if true, the objects are copied before being added to the array
   * @param objects the objects to add to the array
   * @return a new WritableArray containing the objects from the input array
   */
  private static WritableArray getWritableArray(boolean copyObjects, WritableMap... objects) {
    // Create a new WritableArray and add all the objects from the input array to it.
    WritableArray result = Arguments.createArray();
    for (WritableMap object : objects) {
      if (object != null) {
        result.pushMap(copyObjects ? object.copy() : object);
      }
    }
    return result;
  }
}