package com.facebook.react.modules.statusbar;

import android.animation.ArgbEvaluator;
import android.animation.ValueAnimator;
import android.annotation.TargetApi;
import android.app.Activity;
import android.content.Context;
import android.os.Build;
import android.view.View;
import android.view.WindowInsets;
import android.view.WindowInsetsController;
import android.view.WindowManager;
import androidx.annotation.Nullable;
import androidx.core.view.ViewCompat;
import com.facebook.common.logging.FLog;
import com.facebook.fbreact.specs.NativeStatusBarManagerAndroidSpec;
import com.facebook.react.bridge.GuardedRunnable;
import com.facebook.react.bridge.NativeModule;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.UiThreadUtil;
import com.facebook.react.common.MapBuilder;
import com.facebook.react.common.ReactConstants;
import com.facebook.react.module.annotations.ReactModule;
import com.facebook.react.uimanager.PixelUtil;
import java.util.Map;

/**
 * {@link NativeModule} that allows changing the appearance of the status bar.
 */
@ReactModule(name = NativeStatusBarManagerAndroidSpec.NAME)
public class StatusBarModule extends NativeStatusBarManagerAndroidSpec {

  // Constants for exported constants
  private static final String HEIGHT_KEY = "HEIGHT";
  private static final String DEFAULT_BACKGROUND_COLOR_KEY = "DEFAULT_BACKGROUND_COLOR";

  private ReactApplicationContext mContext;
  private Activity mCurrentActivity;

  public StatusBarModule(ReactApplicationContext reactContext) {
    super(reactContext);
    mContext = reactContext;
    mCurrentActivity = getCurrentActivity();
  }

  // Exported constants
  @Override
  public @Nullable Map<String, Object> getTypedExportedConstants() {
    float height = 0f;

    final int heightResId = mContext.getResources().getIdentifier("status_bar_height", "dimen", "android");
    if (heightResId > 0) {
      height = PixelUtil.toDIPFromPixel(mContext.getResources().getDimensionPixelSize(heightResId));
    }

    int statusBarColor = 0;
    if (mCurrentActivity != null) {
      statusBarColor = mCurrentActivity.getWindow().getStatusBarColor();
    }

    return MapBuilder.<String, Object>of(
            HEIGHT_KEY, height,
            DEFAULT_BACKGROUND_COLOR_KEY, String.format("#%06X", (0xFFFFFF & statusBarColor)));
  }

  // Change status bar color
  @Override
  public void setColor(final double colorDouble, final boolean animated) {
    final int color = (int) colorDouble;

    if (mCurrentActivity == null) {
      logError("Ignored status bar change, current activity is null.");
      return;
    }

    runOnUiThread(new Runnable() {
      @Override
      public void run() {
        mCurrentActivity.getWindow().addFlags(WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS);
        if (animated) {
          ValueAnimator colorAnimation = makeColorAnimation(color);

          colorAnimation.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() {
            @Override
            public void onAnimationUpdate(ValueAnimator animator) {
              mCurrentActivity.getWindow().setStatusBarColor((Integer) animator.getAnimatedValue());
            }
          });

          colorAnimation.start();
        } else {
          mCurrentActivity.getWindow().setStatusBarColor(color);
        }
      }
    });
  }

  // Make color animation
  private ValueAnimator makeColorAnimation(int toColor) {
    int fromColor = mCurrentActivity.getWindow().getStatusBarColor();
    ValueAnimator colorAnimation = ValueAnimator.ofObject(new ArgbEvaluator(), fromColor, toColor);
    colorAnimation.setDuration(300).setStartDelay(0);

    return colorAnimation;
  }

  // Hide or show status bar
  @Override
  public void setHidden(final boolean hidden) {
    if (mCurrentActivity == null) {
      logError("Ignored status bar change, current activity is null.");
      return;
    }

    runOnUiThread(new Runnable() {
      @Override
      public void run() {
        if (hidden) {
          mCurrentActivity.getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
          mCurrentActivity.getWindow().clearFlags(WindowManager.LayoutParams.FLAG_FORCE_NOT_FULLSCREEN);
        } else {
          mCurrentActivity.getWindow().addFlags(WindowManager.LayoutParams.FLAG_FORCE_NOT_FULLSCREEN);
          mCurrentActivity.getWindow().clearFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
        }
      }
    });
  }

  // Set status bar translucent or not
  @Override
  public void setTranslucent(final boolean translucent) {
    if (mCurrentActivity == null) {
      logError("Ignored status bar change, current activity is null.");
      return;
    }

    runOnUiThread(new Runnable() {
      @Override
      public void run() {
        // If the status bar is translucent hook into the window insets calculations
        // and consume all the top insets so no padding will be added under the status bar.
        View decorView = mCurrentActivity.getWindow().getDecorView();

        if (translucent) {
          decorView.setOnApplyWindowInsetsListener(new View.OnApplyWindowInsetsListener() {
            @Override
            public WindowInsets onApplyWindowInsets(View v, WindowInsets insets) {
              WindowInsets defaultInsets = v.onApplyWindowInsets(insets);
              return defaultInsets.replaceSystemWindowInsets(
                      defaultInsets.getSystemWindowInsetLeft(),
                      0,
                      defaultInsets.getSystemWindowInsetRight(),
                      defaultInsets.getSystemWindowInsetBottom());
            }
          });
        } else {
          decorView.setOnApplyWindowInsetsListener(null);
        }

        ViewCompat.requestApplyInsets(decorView);
      }
    });
  }

  // Set status bar style (color scheme)
  @Override
  public void setStyle(@Nullable final String style) {
    if (mCurrentActivity == null) {
      logError("Ignored status bar change, current activity is null.");
      return;
    }

    runOnUiThread(new Runnable() {
      @TargetApi(Build.VERSION_CODES.R)
      @Override
      public void run() {
        if (Build.VERSION.SDK_INT > Build.VERSION_CODES.R) {
          setAppearanceUsingWindowInsetsController(style);
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
          setSystemUiVisibilityFlags(style);
        }
      }
    });
  }

  // Set status bar style using WindowInsetsController (API level 30+)
  @TargetApi(Build.VERSION_CODES.R)
  private void setAppearanceUsingWindowInsetsController(String style) {
    WindowInsetsController insetsController = mCurrentActivity.getWindow().getInsetsController();
    if (insetsController == null) {
      return;
    }

    if ("dark-content".equals(style)) {
      // dark-content means dark icons on a light status bar
      insetsController.setSystemBarsAppearance(
              WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS,
              WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS);
    } else {
      insetsController.setSystemBarsAppearance(
              0, WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS);
    }
  }

  // Set status bar style using SystemUiVisibilityFlags (API level 23-29)
  private void setSystemUiVisibilityFlags(String style) {
    View decorView = mCurrentActivity.getWindow().getDecorView();
    int systemUiVisibilityFlags = decorView.getSystemUiVisibility();

    if ("dark-content".equals(style)) {
      systemUiVisibilityFlags |= View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR;
    } else {
      systemUiVisibilityFlags &= ~View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR;
    }

    decorView.setSystemUiVisibility(systemUiVisibilityFlags);
  }

  // Run action on UI thread
  private void runOnUiThread(Runnable action) {
    UiThreadUtil.runOnUiThread(new GuardedRunnable(mContext) {
      @Override
      public void runGuarded() {
        action.run();
      }
    });
  }

  // Log error message
  private void logError(String message) {
    FLog.w(ReactConstants.TAG, "StatusBarModule: " + message);
  }
}

