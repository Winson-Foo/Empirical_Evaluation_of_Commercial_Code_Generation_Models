package com.facebook.react.bridgeless;

import android.app.Activity;
import android.content.Context;
import android.view.View;

import com.facebook.infer.annotation.Nullsafe;
import com.facebook.react.bridge.JSBundleLoader;
import com.facebook.react.bridge.JavaJSExecutor;
import com.facebook.react.bridge.JavaScriptExecutorFactory;
import com.facebook.react.bridge.ReactContext;
import com.facebook.react.bridgeless.internal.bolts.Continuation;
import com.facebook.react.bridgeless.internal.bolts.Task;
import com.facebook.react.devsupport.DevSupportManagerBase;
import com.facebook.react.devsupport.HMRClient;
import com.facebook.react.devsupport.ReactInstanceDevHelper;
import com.facebook.react.devsupport.interfaces.DevSplitBundleCallback;
import com.facebook.react.modules.core.DeviceEventManagerModule;

import javax.annotation.Nullable;

/**
 * An implementation of {@link DevSupportManagerBase} that extends the functionality in
 * {@link DevSupportManagerBase} with some additional, more flexible APIs for asynchronously
 * loading the JS bundle.
 */
@Nullsafe(Nullsafe.Mode.LOCAL)
class BridgelessDevSupportManager extends DevSupportManagerBase {

  private static final String TAG = "Bridgeless";

  private final ReactHost mReactHost;

  public BridgelessDevSupportManager(
      final ReactHost reactHost,
      final Context context,
      @Nullable final String packagerPathForJSBundleName) {
    super(
        context.getApplicationContext(),
        createInstanceDevHelper(reactHost),
        packagerPathForJSBundleName,
        true,
        null,
        null,
        2,
        null,
        null,
        null);
    mReactHost = reactHost;
  }

  @Override
  protected String getUniqueTag() {
    return TAG;
  }

  @Override
  public void loadSplitBundleFromServer(
      final String bundlePath, final DevSplitBundleCallback callback) {
    fetchSplitBundleAndCreateBundleLoader(
        bundlePath,
        new CallbackWithBundleLoader() {
          @Override
          public void onSuccess(JSBundleLoader bundleLoader) {
            mReactHost.loadBundle(bundleLoader)
                .onSuccess(task -> {
                  if (task.getResult()) {
                    String bundleURL = getDevServerHelper().getDevServerSplitBundleURL(bundlePath);
                    final ReactContext reactContext = mReactHost.getCurrentReactContext();
                    if (reactContext != null) {
                      reactContext
                          .getJSModule(HMRClient.class)
                          .registerBundle(bundleURL);
                    }
                    callback.onSuccess();
                  }
                  return null;
                });
          }

          @Override
          public void onError(final String url, final Throwable cause) {
            callback.onError(url, cause);
          }
        });
  }

  @Override
  public void handleReloadJS() {
    hideRedboxDialog();
    mReactHost.reload("BridgelessDevSupportManager.handleReloadJS()");
  }

  private static ReactInstanceDevHelper createInstanceDevHelper(final ReactHost reactHost) {
    return new ReactInstanceDevHelper() {

      @Override
      public void onReloadWithJSDebugger(final JavaJSExecutor.Factory proxyExecutorFactory) {
        // Not implemented
      }

      @Override
      public void onJSBundleLoadedFromServer() {
        throw new IllegalStateException("Not implemented for bridgeless mode");
      }

      @Override
      public void toggleElementInspector() {
        final ReactContext reactContext = reactHost.getCurrentReactContext();
        if (reactContext != null) {
          reactContext
              .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter.class)
              .emit("toggleElementInspector", null);
        }
      }

      @Nullable
      @Override
      public Activity getCurrentActivity() {
        return reactHost.getCurrentActivity();
      }

      @Override
      public JavaScriptExecutorFactory getJavaScriptExecutorFactory() {
        throw new IllegalStateException("Not implemented for bridgeless mode");
      }

      @Nullable
      @Override
      public View createRootView(final String appKey) {
        final Activity currentActivity = getCurrentActivity();
        if (currentActivity != null
            && !reactHost.isSurfaceWithModuleNameAttached(appKey)) {
          final ReactSurface reactSurface =
              ReactSurface.createWithView(currentActivity, appKey, null);
          reactSurface.attach(reactHost);
          reactSurface.start();
          return reactSurface.getView();
        }
        return null;
      }

      @Override
      public void destroyRootView(final View rootView) {
        // Not implemented
      }
    };
  }
} 