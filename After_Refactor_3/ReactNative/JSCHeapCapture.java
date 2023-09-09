package com.facebook.react.devsupport;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.facebook.fbreact.specs.NativeJSCHeapCaptureSpec;
import com.facebook.react.bridge.JavaScriptModule;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.module.annotations.ReactModule;

import java.io.File;

/**
 * Native module for capturing JSC heap snapshots.
 */
@ReactModule(name = JSCHeapCapture.ModuleName, needsEagerInit = true)
public class JSCHeapCapture extends NativeJSCHeapCaptureSpec {

    /**
     * Name of this module.
     */
    public static final String ModuleName = "JSCHeapCapture";

    /**
     * Callback interface for capturing heap snapshots.
     */
    public interface CaptureCallback {
        /**
         * Called when the heap snapshot capture is successful.
         *
         * @param capture The file containing the heap snapshot data.
         */
        void onSuccess(@NonNull File capture);

        /**
         * Called when the heap snapshot capture fails.
         *
         * @param error The exception indicating the failure reason.
         */
        void onFailure(@NonNull CaptureException error);
    }

    /**
     * Exception class for heap snapshot capture failures.
     */
    public static class CaptureException extends Exception {
        /**
         * Constructs a new capture exception with the specified detail message.
         *
         * @param message The detail message.
         */
        public CaptureException(@NonNull String message) {
            super(message);
        }

        /**
         * Constructs a new capture exception with the specified detail message and cause.
         *
         * @param message The detail message.
         * @param cause The cause of the exception.
         */
        public CaptureException(@NonNull String message, @Nullable Throwable cause) {
            super(message, cause);
        }
    }

    /**
     * Callback interface for invoking the JavaScript heap capture function.
     */
    private interface HeapCapture extends JavaScriptModule {
        /**
         * Captures the heap snapshot and saves it to the specified path.
         *
         * @param path The path to save the heap snapshot file.
         */
        void captureHeap(String path);
    }

    /**
     * Manages the progression of heap snapshots capture.
     */
    private static class CaptureProgression {
        /**
         * The callback when heap snapshot capture is in progress.
         */
        private @Nullable CaptureCallback mInProgressCallback;

        /**
         * Checks whether heap snapshot capture is in progress.
         *
         * @return True if heap snapshot capture is in progress, false otherwise.
         */
        public synchronized boolean isInProgress() {
            return mInProgressCallback != null;
        }

        /**
         * Starts heap snapshot capture if it is not already in progress.
         *
         * @param path The folder path to save the heap snapshot file.
         * @param callback The callback when heap snapshot capture is either successful or fails.
         */
        public synchronized void start(@NonNull String path, @NonNull CaptureCallback callback) {
            if (isInProgress()) {
                callback.onFailure(new CaptureException("Heap capture already in progress."));
            } else {
                mInProgressCallback = callback;
                File captureFile = new File(path + "/capture.json");
                captureFile.delete();
                HeapCapture heapCapture = getHeapCapture();
                if (heapCapture != null) {
                    heapCapture.captureHeap(captureFile.getPath());
                } else {
                    callback.onFailure(new CaptureException("Failed to capture heap: Heap capture JavaScript module not registered."));
                    mInProgressCallback = null;
                }
            }
        }

        /**
         * Invoked when the heap snapshot capture is complete.
         *
         * @param path The file path of the heap snapshot file.
         * @param error The error message if the capture fails.
         */
        public synchronized void onComplete(@NonNull String path, @Nullable String error) {
            if (isInProgress()) {
                CaptureCallback callback = mInProgressCallback;
                mInProgressCallback = null;
                if (error == null) {
                    callback.onSuccess(new File(path));
                } else {
                    callback.onFailure(new CaptureException(error));
                }
            }
        }
    }

    /**
     * The progression of heap snapshot capture.
     */
    private final CaptureProgression mCaptureProgression;

    /**
     * Constructs a new JSCHeapCapture instance.
     *
     * @param reactContext The react context.
     */
    public JSCHeapCapture(ReactApplicationContext reactContext) {
        super(reactContext);
        mCaptureProgression = new CaptureProgression();
    }

    /**
     * Captures a heap snapshot and saves it to the specified path.
     * The callback will be called when capture is either successful or failed.
     *
     * @param path The folder path to save the heap snapshot file.
     * @param callback The callback to invoke when capturing heap.
     */
    public void capture(@NonNull String path, @NonNull CaptureCallback callback) {
        mCaptureProgression.start(path, callback);
    }

    /**
     * Invoked when the heap snapshot capture is complete.
     *
     * @param path The file path of the heap snapshot file.
     * @param error The error message if the capture fails.
     */
    @Override
    public void captureComplete(String path, String error) {
        mCaptureProgression.onComplete(path, error);
    }

    /**
     * Gets the JavaScript module for capturing heap snapshots.
     *
     * @return The JavaScript module for capturing heap snapshots, or null if not found.
     */
    private HeapCapture getHeapCapture() {
        ReactApplicationContext reactContext = getReactApplicationContextIfActiveOrWarn();
        if (reactContext != null) {
            return reactContext.getJSModule(HeapCapture.class);
        }
        return null;
    }
}

