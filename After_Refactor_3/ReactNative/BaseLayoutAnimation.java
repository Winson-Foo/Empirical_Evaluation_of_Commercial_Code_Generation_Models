package com.facebook.react.uimanager.layoutanimation;

import android.view.View;
import android.view.animation.Animation;
import android.view.animation.ScaleAnimation;

import com.facebook.react.uimanager.IllegalViewOperationException;

abstract class BaseLayoutAnimation extends AbstractLayoutAnimation {

    enum AnimatedProperty {
        OPACITY,
        SCALE_XY,
        SCALE_X,
        SCALE_Y
    }

    abstract boolean isReverse();

    private static final float FROM_VALUE_REVERSE = 1.0f;
    private static final float TO_VALUE_REVERSE = 0.0f;
    private static final float FROM_VALUE_FORWARD = 0.0f;
    private static final float TO_VALUE_FORWARD = 1.0f;

    private static final float CENTER_RELATIVE = 0.5f;
    private static final float TOP_RELATIVE = 0.0f;
    private static final float BOTTOM_RELATIVE = 1.0f;

    @Override
    boolean isValid() {
        return mDurationMs > 0 && mAnimatedProperty != null;
    }

    @Override
    Animation createAnimationImpl(View view, int x, int y, int width, int height) {
        if (mAnimatedProperty != null) {
            switch (mAnimatedProperty) {
                case OPACITY:
                    float fromValue = isReverse() ? view.getAlpha() : FROM_VALUE_FORWARD;
                    float toValue = isReverse() ? TO_VALUE_REVERSE : view.getAlpha();
                    return new OpacityAnimation(view, fromValue, toValue);
                case SCALE_XY:
                    float fromValueXY = isReverse() ? FROM_VALUE_REVERSE : FROM_VALUE_FORWARD;
                    float toValueXY = isReverse() ? FROM_VALUE_FORWARD : TO_VALUE_FORWARD;
                    return new ScaleAnimation(
                            fromValueXY,
                            toValueXY,
                            fromValueXY,
                            toValueXY,
                            Animation.RELATIVE_TO_SELF,
                            CENTER_RELATIVE,
                            Animation.RELATIVE_TO_SELF,
                            CENTER_RELATIVE);
                case SCALE_X:
                    float fromValueX = isReverse() ? FROM_VALUE_REVERSE : FROM_VALUE_FORWARD;
                    float toValueX = isReverse() ? FROM_VALUE_FORWARD : TO_VALUE_FORWARD;
                    return new ScaleAnimation(
                            fromValueX,
                            toValueX,
                            1f,
                            1f,
                            Animation.RELATIVE_TO_SELF,
                            CENTER_RELATIVE,
                            Animation.RELATIVE_TO_SELF,
                            TOP_RELATIVE);
                case SCALE_Y:
                    float fromValueY = isReverse() ? FROM_VALUE_REVERSE : FROM_VALUE_FORWARD;
                    float toValueY = isReverse() ? FROM_VALUE_FORWARD : TO_VALUE_FORWARD;
                    return new ScaleAnimation(
                            1f,
                            1f,
                            fromValueY,
                            toValueY,
                            Animation.RELATIVE_TO_SELF,
                            TOP_RELATIVE,
                            Animation.RELATIVE_TO_SELF,
                            CENTER_RELATIVE);
                default:
                    throw new IllegalViewOperationException(
                            "Missing animation for property : " + mAnimatedProperty);
            }
        }
        throw new IllegalViewOperationException("Missing animated property from animation config");
    }
} 