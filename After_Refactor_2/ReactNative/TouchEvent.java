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
    
    public static TouchEvent obtain(
            int surfaceId,
            int viewTag,
            TouchEventType touchEventType,
            MotionEvent motionEventToCopy,
            long gestureStartTime,
            float viewX,
            float viewY,
            CoalescingKeyHelper coalescingKeyHelper) {
        TouchEvent event = EVENTS_POOL.acquire();
        if (event == null) {
            event = new TouchEvent();
        }
        event.init(surfaceId, viewTag, touchEventType, Assertions.assertNotNull(motionEventToCopy),
                gestureStartTime, viewX, viewY, coalescingKeyHelper);
        return event;
    }

    private @Nullable MotionEvent motionEvent;
    private @Nullable TouchEventType touchEventType;
    private short coalescingKey;

    private float viewX;
    private float viewY;

    private TouchEvent() {
    }

    private void init(
            int surfaceId,
            int viewTag,
            TouchEventType touchEventType,
            MotionEvent motionEventToCopy,
            long gestureStartTime,
            float viewX,
            float viewY,
            CoalescingKeyHelper coalescingKeyHelper) {
        super.init(surfaceId, viewTag, motionEventToCopy.getEventTime());

        SoftAssertions.assertCondition(
                gestureStartTime != UNSET, "Gesture start time must be initialized");
        short key = 0;
        int action = (motionEventToCopy.getAction() & MotionEvent.ACTION_MASK);
        switch (action) {
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
                key = coalescingKeyHelper.getCoalescingKey(gestureStartTime);
                break;
            case MotionEvent.ACTION_CANCEL:
                coalescingKeyHelper.removeCoalescingKey(gestureStartTime);
                break;
            default:
                throw new RuntimeException("Unhandled MotionEvent action: " + action);
        }
        this.touchEventType = touchEventType;
        this.motionEvent = MotionEvent.obtain(motionEventToCopy);
        this.coalescingKey = key;
        this.viewX = viewX;
        this.viewY = viewY;
    }

    @Override
    public void onDispose() {
        release();
    }

    private void release() {
        if (motionEvent != null) {
            motionEvent.recycle();
            motionEvent = null;
        }

        try {
            EVENTS_POOL.release(this);
        } catch (IllegalStateException e) {
            ReactSoftExceptionLogger.logSoftException(TAG, e);
        }
    }

    @Override
    public String getEventName() {
        return TouchEventType.getJSEventName(Assertions.assertNotNull(touchEventType));
    }

    @Override
    public boolean canCoalesce() {
        switch (Assertions.assertNotNull(touchEventType)) {
            case START:
            case END:
            case CANCEL:
                return false;
            case MOVE:
                return true;
            default:
                throw new RuntimeException("Unknown touch event type: " + touchEventType);
        }
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

    @Override
    protected int getEventCategory() {
        TouchEventType type = touchEventType;
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
        }

        return super.getEventCategory();
    }

    public MotionEvent getMotionEvent() {
        Assertions.assertNotNull(motionEvent);
        return motionEvent;
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

    public TouchEventType getTouchEventType() {
        return Assertions.assertNotNull(touchEventType);
    }

    public float getViewX() {
        return viewX;
    }

    public float getViewY() {
        return viewY;
    }
} 