package com.facebook.react.uimanager.layoutanimation;

import android.view.View;
import android.view.animation.Animation;
import android.view.animation.ScaleAnimation;

import com.facebook.react.uimanager.IllegalViewOperationException;

public abstract class BaseLayoutAnimation extends AbstractLayoutAnimation {

  private static final float MIN_VALUE = 0.0f;
  private static final float MAX_VALUE = 1.0f;
  private static final float CENTER_VALUE = 0.5f;
  private static final float TOP_VALUE = 0.0f;
  private static final float BOTTOM_VALUE = 1.0f;

  abstract boolean isReverse();

  @Override
  boolean isValid() {
    return mDurationMs > 0 && mAnimatedProperty != null;
  }

  @Override
  Animation createAnimationImpl(View view, int x, int y, int width, int height) {
    if (mAnimatedProperty != null) {
      switch (mAnimatedProperty) {
        case OPACITY:
          {
            float fromValue = isReverse() ? view.getAlpha() : MIN_VALUE;
            float toValue = isReverse() ? MIN_VALUE : view.getAlpha();
            return createOpacityAnimation(view, fromValue, toValue);
          }
        case SCALE_XY:
          {
            float fromValue = isReverse() ? MAX_VALUE : MIN_VALUE;
            float toValue = isReverse() ? MIN_VALUE : MAX_VALUE;
            return createScaleAnimation(view, fromValue, toValue, fromValue, toValue, CENTER_VALUE, CENTER_VALUE);
          }
        case SCALE_X:
          {
            float fromValue = isReverse() ? MAX_VALUE : MIN_VALUE;
            float toValue = isReverse() ? MIN_VALUE : MAX_VALUE;
            return createScaleAnimation(view, fromValue, toValue, MAX_VALUE, MAX_VALUE, CENTER_VALUE, TOP_VALUE);
          }
        case SCALE_Y:
          {
            float fromValue = isReverse() ? MAX_VALUE : MIN_VALUE;
            float toValue = isReverse() ? MIN_VALUE : MAX_VALUE;
            return createScaleAnimation(view, MAX_VALUE, MAX_VALUE, fromValue, toValue, TOP_VALUE, CENTER_VALUE);
          }
        default:
          throw new IllegalViewOperationException(
              "Missing animation for property : " + mAnimatedProperty);
      }
    }
    throw new IllegalViewOperationException("Missing animated property from animation config");
  }

  /**
   * Creates an opacity animation.
   *
   * @param view the view to animate
   * @param fromValue the starting value of the animation
   * @param toValue the ending value of the animation
   * @return the opacity animation
   */
  private Animation createOpacityAnimation(View view, float fromValue, float toValue) {
    return new OpacityAnimation(view, fromValue, toValue);
  }

  /**
   * Creates a scale animation.
   *
   * @param view the view to animate
   * @param fromX the starting value of the X-axis scale
   * @param toX the ending value of the X-axis scale
   * @param fromY the starting value of the Y-axis scale
   * @param toY the ending value of the Y-axis scale
   * @param pivotX the pivot point for the X-axis scale
   * @param pivotY the pivot point for the Y-axis scale
   * @return the scale animation
   */
  private Animation createScaleAnimation(View view, float fromX, float toX, float fromY, float toY, float pivotX, float pivotY) {
    return new ScaleAnimation(fromX, toX, fromY, toY, Animation.RELATIVE_TO_SELF, pivotX, Animation.RELATIVE_TO_SELF, pivotY);
  }
}

