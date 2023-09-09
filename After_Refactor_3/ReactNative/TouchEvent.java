package com.facebook.react.uimanager.events;

import android.view.MotionEvent;
import androidx.annotation.Nullable;
import androidx.core.util.Pools;
import com.facebook.infer.annotation.Assertions;
import com.facebook.react.bridge.ReactSoftExceptionLogger;
import com.facebook.react.bridge.SoftAssertions;

public class TouchEvent extends Event<TouchEvent> {
  private static final String TAG = TouchEvent.class.getSimpleName();
  private static final int TOUCH_EVENTS_POOL_SIZE = 3;
  private static final Pools.SynchronizedPool<TouchEvent> EVENTS_POOL =
      new Pools.SynchronizedPool<>(TOUCH_EVENTS_POOL_SIZE);

  public static final long UNSET = Long.MIN_VALUE;

  private @Nullable MotionEvent mMotionEvent;
  private @Nullable TouchEventType mTouchEventType;
  private short mCoalescingKey;
  private float mXInView;
  private float mYInView;

  private TouchEvent() {}

  public static TouchEvent obtain(
      int surfaceId,
      int viewTag,
      TouchEventType touchEventType,
      MotionEvent motionEvent,
      long gestureStartTime,
      float xInView,
      float yInView,
      TouchEventCoalescingKeyHelper coalescingKeyHelper) {
    TouchEvent event = EVENTS_POOL.acquire();
    if (event == null) {
      event = new TouchEvent();
    }
    event.initialize(
        surfaceId, viewTag, touchEventType, motionEvent, gestureStartTime, xInView, yInView, coalescingKeyHelper);
    return event;
  }

  private void initialize(
      int surfaceId,
      int viewTag,
      TouchEventType touchEventType,
      MotionEvent motionEvent,
      long gestureStartTime,
      float xInView,
      float yInView,
      TouchEventCoalescingKeyHelper coalescingKeyHelper) {
    super.initialize(surfaceId, viewTag, motionEvent.getEventTime());

    SoftAssertions.assertCondition(
        gestureStartTime != UNSET, "Gesture start time must be initialized");

    switch (motionEvent.getAction() & MotionEvent.ACTION_MASK) {
      case MotionEvent.ACTION_DOWN:
        coalescingKeyHelper.addCoalescingKey(gestureStartTime);
        break;
      case MotionEvent.ACTION_UP:
        coalescingKeyHelper.removeCoalescingKey(gestureStartTime);
        break;
      case MotionEvent.ACTION_POINTER_DOWN:
      case MotionEvent.ACTION_POINTER_UP:
        coalescingKeyHelper.incrementCoalescingKey(gestureStartTime);
        break;
      case MotionEvent.ACTION_MOVE:
        mCoalescingKey = coalescingKeyHelper.getCoalescingKey(gestureStartTime);
        break;
      case MotionEvent.ACTION_CANCEL:
        coalescingKeyHelper.removeCoalescingKey(gestureStartTime);
        break;
      default:
        throw new RuntimeException("Unhandled MotionEvent action: " + motionEvent.getAction());
    }

    mTouchEventType = touchEventType;
    mMotionEvent = MotionEvent.obtain(motionEvent);
    mXInView = xInView;
    mYInView = yInView;
  }

  public static TouchEvent obtain(
      int viewTag,
      TouchEventType touchEventType,
      MotionEvent motionEvent,
      long gestureStartTime,
      float xInView,
      float yInView,
      TouchEventCoalescingKeyHelper coalescingKeyHelper) {
    return obtain(-1, viewTag, touchEventType, motionEvent, gestureStartTime, xInView, yInView, coalescingKeyHelper);
  }

  @Override
  public void onDispose() {
    MotionEvent motionEvent = mMotionEvent;
    mMotionEvent = null;
    if (motionEvent != null) {
      motionEvent.recycle();
    }

    try {
      EVENTS_POOL.release(this);
    } catch (IllegalStateException e) {
      ReactSoftExceptionLogger.logSoftException(TAG, e);
    }
  }

  @Override
  public String getEventName() {
    return TouchEventType.getJSEventName(Assertions.assertNotNull(mTouchEventType));
  }

  @Override
  public boolean canCoalesce() {
    switch (Assertions.assertNotNull(mTouchEventType)) {
      case START:
      case END:
      case CANCEL:
        return false;
      case MOVE:
        return true;
      default:
        throw new RuntimeException("Unknown touch event type: " + mTouchEventType);
    }
  }

  @Override
  public short getCoalescingKey() {
    return mCoalescingKey;
  }

  @Override
  public void dispatch(RCTEventEmitter rctEventEmitter) {
    if (verifyMotionEvent()) {
      TouchesHelper.sendTouchesLegacy(rctEventEmitter, this);
    }
  }

  @Override
  public void dispatchModern(RCTModernEventEmitter rctEventEmitter) {
    if (verifyMotionEvent()) {
      rctEventEmitter.receiveTouches(this);
    }
  }

  @Override
  protected int getEventCategory() {
    TouchEventType type = mTouchEventType;
    if (type == null) {
      return EventCategoryDef.UNSPECIFIED;
    }
    switch (type) {
      case START:
        return EventCategoryDef.CONTINUOUS_START;
      case END:
      case CANCEL:
        return EventCategoryDef.CONTINUOUS_END;
      case MOVE:
        return EventCategoryDef.CONTINUOUS;
      default:
        throw new RuntimeException("Unknown touch event type: " + type);
    }
  }

  public boolean verifyMotionEvent() {
    if (mMotionEvent == null) {
      ReactSoftExceptionLogger.logSoftException(
          TAG,
          new IllegalStateException(
              "Cannot dispatch a TouchEvent that has no MotionEvent; the TouchEvent has been recycled"));
      return false;
    }
    return true;
  }

  public MotionEvent getMotionEvent() {
    Assertions.assertNotNull(mMotionEvent);
    return mMotionEvent;
  }

  public TouchEventType getTouchEventType() {
    return Assertions.assertNotNull(mTouchEventType);
  }

  public float getxInView() {
    return mXInView;
  }

  public float getyInView() {
    return mYInView;
  }
} 