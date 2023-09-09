// Interface to report success or failure of a task.
public interface TaskCompletionListener {
  void onSuccess();
  void onFailure(Exception cause);
}

// Interface to track progress for a task.
public interface ProgressListener {
  void onProgress(String status, int done, int total);
}

// Interface to download dev bundle from server.
public interface DevBundleDownloader {
  void downloadBundle(String url, TaskCompletionListener listener, ProgressListener progressListener);
}

// Implement this interface to display progress while downloading a bundle.
public interface DevBundleDownloadProgressView {
  void showProgress(String status, int percentDone);
  void hideProgress();
}