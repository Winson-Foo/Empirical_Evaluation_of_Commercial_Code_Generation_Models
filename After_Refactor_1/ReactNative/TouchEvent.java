package com.facebook.react.uimanager.events;

import android.view.MotionEvent;
import androidx.annotation.NonNull;
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

  private final MotionEvent motionEvent;
  private final TouchEventType touchEventType;
  private final short coalescingKey;
  private final float viewX;
  private final float viewY;

  private TouchEvent(
      int surfaceId,
      int viewTag,
      MotionEvent motionEvent,
      TouchEventType touchEventType,
      short coalescingKey,
      float viewX,
      float viewY) {
    super(surfaceId, viewTag, motionEvent.getEventTime());
    this.motionEvent = motionEvent;
    this.touchEventType = touchEventType;
    this.coalescingKey = coalescingKey;
    this.viewX = viewX;
    this.viewY = viewY;
  }

  public static Builder builder() {
    return new Builder();
  }

  public MotionEvent getMotionEvent() {
    Assertions.assertNotNull(motionEvent);
    return motionEvent;
  }

  public TouchEventType getTouchEventType() {
    return Assertions.assertNotNull(touchEventType);
  }

  public float getViewX() {
    return viewX;
  }

  public float getViewY() {
    return viewY;
  }

  @Override
  public String getEventName() {
    return TouchEventType.getJSEventName(Assertions.assertNotNull(touchEventType));
  }

  @Override
  public boolean canCoalesce() {
    return touchEventType == TouchEventType.MOVE;
  }

  @Override
  public short getCoalescingKey() {
    return coalescingKey;
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

  private boolean verifyMotionEvent() {
    if (motionEvent == null) {
      ReactSoftExceptionLogger.logSoftException(
          TAG,
          new IllegalStateException(
              "Cannot dispatch a TouchEvent that has no MotionEvent; the TouchEvent has been recycled"));
      return false;
    }
    return true;
  }

  private static short calculateCoalescingKey(
      MotionEvent motionEvent,
      long gestureStartTime,
      TouchEventCoalescingKeyHelper touchEventCoalescingKeyHelper) {
    int action = motionEvent.getActionMasked();
    switch (action) {
      case MotionEvent.ACTION_DOWN:
        touchEventCoalescingKeyHelper.addCoalescingKey(gestureStartTime);
        break;
      case MotionEvent.ACTION_UP:
        touchEventCoalescingKeyHelper.removeCoalescingKey(gestureStartTime);
        break;
      case MotionEvent.ACTION_POINTER_DOWN:
      case MotionEvent.ACTION_POINTER_UP:
        touchEventCoalescingKeyHelper.incrementCoalescingKey(gestureStartTime);
        break;
      case MotionEvent.ACTION_MOVE:
        return touchEventCoalescingKeyHelper.getCoalescingKey(gestureStartTime);
      case MotionEvent.ACTION_CANCEL:
        touchEventCoalescingKeyHelper.removeCoalescingKey(gestureStartTime);
        break;
      default:
        throw new RuntimeException("Unhandled MotionEvent action: " + action);
    }
    return 0;
  }

  public static class Builder {
    private MotionEvent motionEvent;
    private TouchEventType touchEventType;
    private short coalescingKey;
    private float viewX;
    private float viewY;
    private int surfaceId;
    private int viewTag;
    private long gestureStartTime;
    private TouchEventCoalescingKeyHelper touchEventCoalescingKeyHelper;

    public Builder withMotionEvent(@NonNull MotionEvent motionEvent) {
      this.motionEvent = motionEvent;
      return this;
    }

    public Builder withTouchEventType(@NonNull TouchEventType touchEventType) {
      this.touchEventType = touchEventType;
      return this;
    }

    public Builder withSurfaceId(int surfaceId) {
      this.surfaceId = surfaceId;
      return this;
    }

    public Builder withViewTag(int viewTag) {
      this.viewTag = viewTag;
      return this;
    }

    public Builder withGestureStartTime(long gestureStartTime) {
      this.gestureStartTime = gestureStartTime;
      return this;
    }

    public Builder withViewCoordinates(float viewX, float viewY) {
      this.viewX = viewX;
      this.viewY = viewY;
      return this;
    }

    @Nullable
    public TouchEvent build() {
      if (motionEvent == null || touchEventType == null) {
        return null;
      }

      coalescingKey =
          calculateCoalescingKey(motionEvent, gestureStartTime, touchEventCoalescingKeyHelper);

      TouchEvent event = EVENTS_POOL.acquire();
      if (event == null) {
        event = new TouchEvent(surfaceId, viewTag, motionEvent, touchEventType, coalescingKey, viewX, viewY);
      } else {
        event.init(surfaceId, viewTag, motionEvent.getEventTime());
        event.motionEvent = motionEvent;
        event.touchEventType = touchEventType;
        event.coalescingKey = coalescingKey;
        event.viewX = viewX;
        event.viewY = viewY;
      }
      return event;
    }
  }
}