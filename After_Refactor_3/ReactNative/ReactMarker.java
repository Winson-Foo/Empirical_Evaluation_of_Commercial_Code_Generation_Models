/**
 * Static class that allows markers to be placed in React code and responded to in a configurable
 * way
 */
public class ReactMarker {

  /**
   * Listener for React markers.
   */
  public interface MarkerListener {
    /**
     * Logs a React marker.
     */
    void logMarker(ReactMarkerConstants name, @Nullable String tag, int instanceKey);
  }

  /**
   * Listener for Fabric-specific markers.
   */
  public interface FabricMarkerListener {
    /**
     * Logs a Fabric-specific marker.
     */
    void logFabricMarker(
        ReactMarkerConstants name, @Nullable String tag, int instanceKey, long timestamp);
  }

  // Use a set here to avoid duplicate entries.
  private static final Set<MarkerListener> sListeners = new CopyOnWriteArraySet<>();

  // Use a set here to avoid duplicate entries.
  private static final Set<FabricMarkerListener> sFabricMarkerListeners = new CopyOnWriteArraySet<>();

  // The android app start time that to be set by the corresponding app
  private static long sAppStartTime;

  /**
   * Adds a listener for React markers.
   */
  public static void addMarkerListener(MarkerListener listener) {
    if (!sListeners.contains(listener)) {
      sListeners.add(listener);
    }
  }

  /**
   * Removes a listener for React markers.
   */
  public static void removeMarkerListener(MarkerListener listener) {
    sListeners.remove(listener);
  }

  /**
   * Clears all listeners for React markers.
   */
  public static void clearMarkerListeners() {
    sListeners.clear();
  }

  /**
   * Adds a listener for Fabric-specific markers.
   */
  public static void addFabricMarkerListener(FabricMarkerListener listener) {
    if (!sFabricMarkerListeners.contains(listener)) {
      sFabricMarkerListeners.add(listener);
    }
  }

  /**
   * Removes a listener for Fabric-specific markers.
   */
  public static void removeFabricMarkerListener(FabricMarkerListener listener) {
    sFabricMarkerListeners.remove(listener);
  }

  /**
   * Clears all listeners for Fabric-specific markers.
   */
  public static void clearFabricMarkerListeners() {
    sFabricMarkerListeners.clear();
  }

  /**
   * Logs a React marker with the given name and tag.
   */
  public static void logMarker(String name, @Nullable String tag) {
    logMarker(name, tag, 0);
  }

  /**
   * Logs a React marker with the given name, tag, and instance key.
   */
  public static void logMarker(String name, @Nullable String tag, int instanceKey) {
    ReactMarkerConstants marker = ReactMarkerConstants.valueOf(name);
    logMarker(marker, tag, instanceKey);
  }

  /**
   * Logs a React marker with the given constant.
   */
  public static void logMarker(ReactMarkerConstants name) {
    logMarker(name, null, 0);
  }

  /**
   * Logs a React marker with the given constant and instance key.
   */
  public static void logMarker(ReactMarkerConstants name, int instanceKey) {
    logMarker(name, null, instanceKey);
  }

  /**
   * Logs a React marker with the given constant, tag, and instance key.
   */
  public static void logMarker(ReactMarkerConstants name, @Nullable String tag, int instanceKey) {
    logFabricMarker(name, tag, instanceKey);
    for (MarkerListener listener : sListeners) {
      listener.logMarker(name, tag, instanceKey);
    }
  }

  /**
   * Logs a Fabric-specific marker with the given constant, tag, instance key, and timestamp.
   */
  public static void logFabricMarker(
      ReactMarkerConstants name, @Nullable String tag, int instanceKey, long timestamp) {
    for (FabricMarkerListener listener : sFabricMarkerListeners) {
      listener.logFabricMarker(name, tag, instanceKey, timestamp);
    }
  }

  /**
   * Logs a Fabric-specific marker with the given constant, tag, and instance key, using the
   * current timestamp.
   */
  public static void logFabricMarker(
      ReactMarkerConstants name, @Nullable String tag, int instanceKey) {
    logFabricMarker(name, tag, instanceKey, SystemClock.uptimeMillis());
  }

  /**
   * Sets the app start time.
   */
  public static void setAppStartTime(long appStartTime) {
    sAppStartTime = appStartTime;
  }

  /**
   * Gets the app start time.
   */
  public static double getAppStartTime() {
    return (double) sAppStartTime;
  }
} 