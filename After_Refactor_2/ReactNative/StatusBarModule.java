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
    private static final String HEIGHT_KEY = "HEIGHT";
    private static final String DEFAULT_BACKGROUND_COLOR_KEY = "DEFAULT_BACKGROUND_COLOR";
    public static final String NAME = "StatusBarManager";
    private Context context;
    private Activity activity;

    private float calculateHeight() {
        final int heightResId =
                context.getResources().getIdentifier("status_bar_height", "dimen", "android");
        return heightResId > 0 ? PixelUtil.toDIPFromPixel(context.getResources().getDimensionPixelSize(heightResId))
                : 0;
    }

    private String getStatusBarColor() {
        String statusBarColorString = "black";
        if (activity != null) {
            final int statusBarColor = activity.getWindow().getStatusBarColor();
            statusBarColorString = String.format("#%06X", (0xFFFFFF & statusBarColor));
        }
        return statusBarColorString;
    }

    private void handleStatusBarChange(int color, boolean animated) {
        activity.getWindow().addFlags(WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS);
        if (animated) {
            int curColor = activity.getWindow().getStatusBarColor();
            ValueAnimator colorAnimation = ValueAnimator.ofObject(new ArgbEvaluator(), curColor, color);

            colorAnimation.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() {
                @Override
                public void onAnimationUpdate(ValueAnimator animator) {
                    activity.getWindow().setStatusBarColor((Integer) animator.getAnimatedValue());
                }
            });

            colorAnimation.setDuration(300).setStartDelay(0);
            colorAnimation.start();
        } else {
            activity.getWindow().setStatusBarColor(color);
        }
    }

    private View.OnApplyWindowInsetsListener onApplyWindowInsetsListener(boolean translucent) {
        return translucent ? new View.OnApplyWindowInsetsListener() {
            @Override
            public WindowInsets onApplyWindowInsets(View v, WindowInsets insets) {
                WindowInsets defaultInsets = v.onApplyWindowInsets(insets);
                return defaultInsets.replaceSystemWindowInsets(
                        defaultInsets.getSystemWindowInsetLeft(), 0, defaultInsets.getSystemWindowInsetRight()
                        , defaultInsets.getSystemWindowInsetBottom());
            }
        } : null;
    }

    private void setSystemBarsAppearance(String style) {
        if (Build.VERSION.SDK_INT > Build.VERSION_CODES.R) {
            WindowInsetsController insetsController = activity.getWindow().getInsetsController();
            if ("dark-content".equals(style)) {
                insetsController.setSystemBarsAppearance(WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS,
                        WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS);
            } else {
                insetsController.setSystemBarsAppearance(0, WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS);
            }
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            View decorView = activity.getWindow().getDecorView();
            int systemUiVisibilityFlags = decorView.getSystemUiVisibility();
            if ("dark-content".equals(style)) {
                systemUiVisibilityFlags |= View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR;
            } else {
                systemUiVisibilityFlags &= ~View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR;
            }
            decorView.setSystemUiVisibility(systemUiVisibilityFlags);
        }
    }

    public StatusBarModule(ReactApplicationContext reactContext) {
        super(reactContext);
    }

    @Override
    public void initialize() {
        super.initialize();
        context = getReactApplicationContext();
        activity = getCurrentActivity();
    }

    @Override
    public @Nullable Map<String, Object> getTypedExportedConstants() {
        return MapBuilder.<String, Object>of(
                HEIGHT_KEY, calculateHeight(),
                DEFAULT_BACKGROUND_COLOR_KEY, getStatusBarColor());
    }

    @Override
    public void setColor(final double colorDouble, final boolean animated) {
        if (activity == null) {
            FLog.w(
                    ReactConstants.TAG,
                    "StatusBarModule: Ignored status bar change, current activity is null.");
            return;
        }

        UiThreadUtil.runOnUiThread(
                new GuardedRunnable(getReactApplicationContext()) {
                    @Override
                    public void runGuarded() {
                        handleStatusBarChange((int) colorDouble, animated);
                    }
                });
    }

    @Override
    public void setTranslucent(final boolean translucent) {
        if (activity == null) {
            FLog.w(
                    ReactConstants.TAG,
                    "StatusBarModule: Ignored status bar change, current activity is null.");
            return;
        }

        UiThreadUtil.runOnUiThread(
                new GuardedRunnable(getReactApplicationContext()) {
                    @Override
                    public void runGuarded() {
                        final View decorView = activity.getWindow().getDecorView();
                        decorView.setOnApplyWindowInsetsListener(onApplyWindowInsetsListener(translucent));
                        ViewCompat.requestApplyInsets(decorView);
                    }
                });
    }

    @Override
    public void setHidden(final boolean hidden) {
        if (activity == null) {
            FLog.w(
                    ReactConstants.TAG,
                    "StatusBarModule: Ignored status bar change, current activity is null.");
            return;
        }
        UiThreadUtil.runOnUiThread(
                new Runnable() {
                    @Override
                    public void run() {
                        if (hidden) {
                            activity.getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
                            activity.getWindow().clearFlags(WindowManager.LayoutParams.FLAG_FORCE_NOT_FULLSCREEN);
                        } else {
                            activity.getWindow().addFlags(WindowManager.LayoutParams.FLAG_FORCE_NOT_FULLSCREEN);
                            activity.getWindow().clearFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
                        }
                    }
                });
    }

    @Override
    public void setStyle(@Nullable final String style) {
        if (activity == null) {
            FLog.w(
                    ReactConstants.TAG,
                    "StatusBarModule: Ignored status bar change, current activity is null.");
            return;
        }

        UiThreadUtil.runOnUiThread(new Runnable() {
            @TargetApi(Build.VERSION_CODES.R)
            @Override
            public void run() {
                setSystemBarsAppearance(style);
            }
        });
    }
} 