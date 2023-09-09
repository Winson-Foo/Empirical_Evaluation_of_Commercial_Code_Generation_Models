package com.facebook.react.uimanager.layoutanimation;

import android.view.View;
import android.view.animation.Animation;
import android.view.animation.ScaleAnimation;

import com.facebook.react.uimanager.IllegalViewOperationException;

/** Class responsible for default layout animation, i.e animation of view creation and deletion. */
/* package */ abstract class BaseLayoutAnimation extends AbstractLayoutAnimation {

  enum AnimatedProperty {
    OPACITY,
    SCALE_XY,
    SCALE_X,
    SCALE_Y,
  }

  abstract boolean isReverse();

  private static final float FROM_VALUE = isReverse() ? 1.0f : 0.0f;
  private static final float TO_VALUE = isReverse() ? 0.0f : 1.0f;

  @Override
  boolean isValid() {
    return mDurationMs > 0 && mAnimatedProperty != null;
  }

  @Override
  Animation createAnimationForProperty(View view, AnimatedProperty property) {
    switch (property) {
      case OPACITY:
        return createOpacityAnimation(view);
      case SCALE_XY:
        return createScaleXYAnimation(view);
      case SCALE_X:
        return createScaleXAnimation(view);
      case SCALE_Y:
        return createScaleYAnimation(view);
      default:
        throw new IllegalViewOperationException(
            "Missing animation for property : " + mAnimatedProperty);
    }
  }

  private Animation createOpacityAnimation(View view) {
    return new OpacityAnimation(view, FROM_VALUE, TO_VALUE);
  }

  private Animation createScaleXYAnimation(View view) {
    return new ScaleAnimation(
        FROM_VALUE,
        TO_VALUE,
        FROM_VALUE,
        TO_VALUE,
        Animation.RELATIVE_TO_SELF,
        .5f,
        Animation.RELATIVE_TO_SELF,
        .5f);
  }

  private Animation createScaleXAnimation(View view) {
    return new ScaleAnimation(
        FROM_VALUE,
        TO_VALUE,
        1f,
        1f,
        Animation.RELATIVE_TO_SELF,
        .5f,
        Animation.RELATIVE_TO_SELF,
        0f);
  }

  private Animation createScaleYAnimation(View view) {
    return new ScaleAnimation(
        1f,
        1f,
        FROM_VALUE,
        TO_VALUE,
        Animation.RELATIVE_TO_SELF,
        0f,
        Animation.RELATIVE_TO_SELF,
        .5f);
  }
}