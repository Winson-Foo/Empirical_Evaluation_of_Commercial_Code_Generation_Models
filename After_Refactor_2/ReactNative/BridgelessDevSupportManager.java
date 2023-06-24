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

@Nullsafe(Nullsafe.Mode.LOCAL)
class BridgelessDevSupportManager extends DevSupportManagerBase {

    private final ReactHost mReactHost;

    public BridgelessDevSupportManager(
        ReactHost host, Context context, @Nullable String packagerPathForJSBundleName) {
        super(
            context.getApplicationContext(),
            createInstanceDevHelper(host),
            packagerPathForJSBundleName,
            true /* enableOnCreate */,
            null /* redBoxHandler */,
            null /* devBundleDownloadListener */,
            2 /* minNumShakes */,
            null /* customPackagerCommandHandlers */,
            null /* surfaceDelegateFactory */,
            null /* devLoadingViewManager */);
        mReactHost = host;
    }

    @Override
    protected String getUniqueTag() {
        return "Bridgeless";
    }

    @Override
    public void loadSplitBundleFromServer(
        final String bundlePath, final DevSplitBundleCallback bundleCallback) {
        fetchSplitBundleAndCreateBundleLoader(
            bundlePath, new CallbackWithBundleLoader(bundleCallback));
    }

    @Override
    public void handleReloadJS() {
        hideRedboxDialog();
        mReactHost.reload("BridgelessDevSupportManager.handleReloadJS()");
    }

    private static ReactInstanceDevHelper createInstanceDevHelper(final ReactHost reactHost) {
        return new BridgelessReactInstanceDevHelper(reactHost);
    }
}

class CallbackWithBundleLoader implements FetchBundleCallback {
    private final DevSplitBundleCallback bundleCallback;

    public CallbackWithBundleLoader(DevSplitBundleCallback bundleCallback) {
        this.bundleCallback = bundleCallback;
    }

    @Override
    public void onSuccess(final JSBundleLoader bundleLoader) {
        mReactHost
            .loadBundle(bundleLoader)
            .onSuccess(new HandleBooleanResultContinuation(bundleLoader));
    }

    @Override
    public void onError(String url, Throwable cause) {
        bundleCallback.onError(url, cause);
    }
}

class HandleBooleanResultContinuation implements Continuation<Boolean, Void> {
    private final JSBundleLoader bundleLoader;

    public HandleBooleanResultContinuation(JSBundleLoader bundleLoader) {
        this.bundleLoader = bundleLoader;
    }

    @Override
    public Void then(Task<Boolean> task) {
        if (task.getResult().equals(Boolean.TRUE)) {
            String bundleURL =
                getDevServerHelper().getDevServerSplitBundleURL(bundlePath);
            ReactContext reactContext = mReactHost.getCurrentReactContext();
            if (reactContext != null) {
                reactContext.getJSModule(HMRClient.class).registerBundle(bundleURL);
            }
            bundleCallback.onSuccess();
        }
        return null;
    }
}

class BridgelessReactInstanceDevHelper implements ReactInstanceDevHelper {
    private final ReactHost reactHost;

    public BridgelessReactInstanceDevHelper(final ReactHost reactHost) {
        this.reactHost = reactHost;
    }

    @Override
    public void onReloadWithJSDebugger(JavaJSExecutor.Factory proxyExecutorFactory) {
        // Not implemented
    }

    @Override
    public void onJSBundleLoadedFromServer() {
        throw new IllegalStateException("Not implemented for bridgeless mode");
    }

    @Override
    public void toggleElementInspector() {
        ReactContext reactContext = reactHost.getCurrentReactContext();
        if (reactContext != null) {
            reactContext
                .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter.class)
                .emit("toggleElementInspector", null);
        }
    }

    @androidx.annotation.Nullable
    @Override
    public Activity getCurrentActivity() {
        return reactHost.getCurrentActivity();
    }

    @Override
    public JavaScriptExecutorFactory getJavaScriptExecutorFactory() {
        throw new IllegalStateException("Not implemented for bridgeless mode");
    }

    @androidx.annotation.Nullable
    @Override
    public View createRootView(String appKey) {
        Activity currentActivity = getCurrentActivity();
        if (currentActivity != null && !reactHost.isSurfaceWithModuleNameAttached(appKey)) {
            ReactSurface reactSurface = ReactSurface.createWithView(currentActivity, appKey, null);
            reactSurface.attach(reactHost);
            reactSurface.start();

            return reactSurface.getView();
        }
        return null;
    }

    @Override
    public void destroyRootView(View rootView) {
        // Not implemented
    }
}

interface FetchBundleCallback {
    void onSuccess(JSBundleLoader bundleLoader);

    void onError(String url, Throwable cause);
}

