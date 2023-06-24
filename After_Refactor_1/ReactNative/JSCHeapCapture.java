public class JSCHeapCapture extends NativeJSCHeapCaptureSpec {

    private static final String ERROR_CAPTURE_IN_PROGRESS = "Heap capture already in progress.";
    private static final String ERROR_MODULE_NOT_REGISTERED = "Heap capture js module not registered.";

    public interface HeapCapture extends JavaScriptModule {
        void captureHeap(@NonNull String path);
    }

    public static class CaptureException extends Exception {
        CaptureException(@NonNull String message) {
            super(message);
        }

        CaptureException(@NonNull String message, @Nullable Throwable cause) {
            super(message, cause);
        }
    }

    private @Nullable CaptureCallback mCaptureInProgress;

    public JSCHeapCapture(@NonNull ReactApplicationContext reactContext) {
        super(reactContext);
        mCaptureInProgress = null;
    }

    public synchronized void captureHeap(String path, final CaptureCallback callback) {
        if (mCaptureInProgress != null) {
            callback.onFailure(new CaptureException(ERROR_CAPTURE_IN_PROGRESS));
            return;
        }

        File f = new File(path + "/capture.json");
        f.delete();

        ReactApplicationContext reactApplicationContext = getReactApplicationContextIfActiveOrWarn();
        if (reactApplicationContext == null) {
            callback.onFailure(new CaptureException("ReactApplication is not active"));
            return;
        }

        HeapCapture heapCapture = reactApplicationContext.getJSModule(HeapCapture.class);
        if (heapCapture == null) {
            callback.onFailure(new CaptureException(ERROR_MODULE_NOT_REGISTERED));
            return;
        }

        mCaptureInProgress = callback;
        heapCapture.captureHeap(f.getPath());
    }

    @Override
    public synchronized void captureComplete(String path, String error) {
        if (mCaptureInProgress == null) {
            return;
        }

        if (error == null) {
            File file = new File(path);
            if (file.exists()) {
                mCaptureInProgress.onSuccess(file);
            } else {
                mCaptureInProgress.onFailure(new CaptureException("Heap capture file not found"));
            }
        } else {
            mCaptureInProgress.onFailure(new CaptureException(error));
        }

        mCaptureInProgress = null;
    }
}