package com.facebook.react.devsupport;

import android.os.Handler;
import android.os.Looper;
import androidx.annotation.Nullable;
import com.facebook.fbreact.specs.NativeJSCHeapCaptureSpec;
import com.facebook.react.bridge.JavaScriptModule;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.module.annotations.ReactModule;

import java.io.File;
import java.util.Objects;

/**
 * A module that provides the ability to capture a snapshot of the JavaScript heap on demand.
 */
@ReactModule(name = NativeJSCHeapCaptureSpec.NAME, needsEagerInit = true)
public class JscHeapCaptureModule extends NativeJSCHeapCaptureSpec {

  /**
   * An interface for invoking the JavaScript method responsible for capturing the heap.
   */
  public interface JscHeapCaptureInterface extends JavaScriptModule {
    void captureHeap(String path);
  }

  /**
   * An exception that can be thrown when heap capture fails.
   */
  public static class CaptureException extends Exception {
    CaptureException(String message) {
      super(message);
    }

    CaptureException(String message, Throwable cause) {
      super(message, cause);
    }
  }

  /**
   * A callback to be invoked when heap capture completes.
   */
  public interface CaptureCallback {
    /**
     * Invoked when heap capture succeeds.
     *
     * @param capture the file containing the heap capture.
     */
    void onSuccess(File capture);

    /**
     * Invoked when heap capture fails.
     *
     * @param error the exception that caused the failure.
     */
    void onFailure(CaptureException error);
  }

  private final Handler mHandler;
  private @Nullable CaptureCallback mCaptureInProgress;

  public JscHeapCaptureModule(ReactApplicationContext reactContext) {
    super(reactContext);
    mHandler = new Handler(Looper.getMainLooper());
    mCaptureInProgress = null;
  }

  /**
   * Captures the JavaScript heap and writes the result to a file.
   *
   * @param path the directory in which to save the heap capture.
   * @param callback a callback to be invoked when heap capture completes.
   * @throws NullPointerException if {@code path} or {@code callback} is null.
   */
  public synchronized void captureHeap(String path, final CaptureCallback callback) {
    Objects.requireNonNull(path);
    Objects.requireNonNull(callback);

    if (mCaptureInProgress != null) {
      callback.onFailure(new CaptureException("Heap capture already in progress."));
      return;
    }

    final File f = new File(path + "/capture.json");
    f.delete();

    ReactApplicationContext reactContext = getReactApplicationContextIfActiveOrWarn();

    if (reactContext != null) {
      JscHeapCaptureInterface heapCapture = reactContext.getJSModule(JscHeapCaptureInterface.class);
      if (heapCapture == null) {
        callback.onFailure(new CaptureException("Heap capture js module not registered."));
        return;
      }

      mCaptureInProgress = callback;
      heapCapture.captureHeap(f.getPath());
    }
  }

  @Override
  public synchronized void captureComplete(String path, String error) {
    mHandler.post(() -> {
      if (mCaptureInProgress != null) {
        if (error == null) {
          mCaptureInProgress.onSuccess(new File(path));
        } else {
          mCaptureInProgress.onFailure(new CaptureException(error));
        }

        mCaptureInProgress = null;
      }
    });
  }
}