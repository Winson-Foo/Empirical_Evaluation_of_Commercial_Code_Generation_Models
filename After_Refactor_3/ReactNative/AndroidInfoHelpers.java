// AndroidInfoHelpers.java

package com.facebook.react.modules.systeminfo;

import android.content.Context;
import android.content.res.Resources;
import android.os.Build;
import com.facebook.common.logging.FLog;
import com.facebook.react.R;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.Locale;

/**
 * Helper methods for retrieving information about the Android environment.
 */
public class AndroidInfoHelpers {

  // IP addresses for different environments
  private static final String EMULATOR_LOCALHOST = "10.0.2.2";
  private static final String GENYMOTION_LOCALHOST = "10.0.3.2";
  private static final String DEVICE_LOCALHOST = "localhost";

  // Name of the system property that specifies the Metro host
  private static final String METRO_HOST_PROP_NAME = "metro.host";

  // Tag for logging messages
  private static final String TAG = AndroidInfoHelpers.class.getSimpleName();

  /**
   * Returns whether the app is running on a Genymotion virtual device.
   */
  public static boolean isRunningOnGenymotion() {
    return Build.FINGERPRINT.contains("vbox");
  }

  /**
   * Returns whether the app is running on the stock Android emulator.
   */
  public static boolean isRunningOnStockEmulator() {
    return Build.FINGERPRINT.contains("generic")
        || Build.FINGERPRINT.startsWith("google/sdk_gphone");
  }

  /**
   * Returns the IP address of the development server.
   *
   * @param port the port number of the development server
   * @return the IP address of the development server
   */
  public static String getServerHost(int port) {
    return getServerIpAddress(port);
  }

  /**
   * Returns the IP address of the development server.
   *
   * @param context the app context
   * @return the IP address of the development server
   */
  public static String getServerHost(Context context) {
    int port = getIntegerResource(context, R.integer.react_native_dev_server_port);
    return getServerIpAddress(port);
  }

  /**
   * Returns the adb command for reverse port forwarding.
   *
   * @param port the port to forward
   * @return the adb command for reverse port forwarding
   */
  public static String getAdbReverseTcpCommand(int port) {
    return "adb reverse tcp:" + port + " tcp:" + port;
  }

  /**
   * Returns the adb command for reverse port forwarding.
   *
   * @param context the app context
   * @return the adb command for reverse port forwarding
   */
  public static String getAdbReverseTcpCommand(Context context) {
    int port = getIntegerResource(context, R.integer.react_native_dev_server_port);
    return getAdbReverseTcpCommand(port);
  }

  /**
   * Returns the IP address of the inspector proxy server.
   *
   * @param context the app context
   * @return the IP address of the inspector proxy server
   */
  public static String getInspectorProxyHost(Context context) {
    int port = getIntegerResource(context, R.integer.react_native_inspector_proxy_port);
    return getServerIpAddress(port);
  }

  /**
   * Returns a friendly name for the current device.
   */
  public static String getFriendlyDeviceName() {
    if (isRunningOnGenymotion()) {
      // Genymotion already has a friendly name by default
      return Build.MODEL;
    } else {
      return Build.MODEL + " - " + Build.VERSION.RELEASE + " - API " + Build.VERSION.SDK_INT;
    }
  }

  /**
   * Returns the value of the "metro.host" system property.
   */
  private static synchronized String getMetroHostPropValue() {
    if (metroHostPropValue != null) {
      return metroHostPropValue;
    }
    Process process = null;
    BufferedReader reader = null;
    try {
      process =
          Runtime.getRuntime().exec(new String[] {"/system/bin/getprop", METRO_HOST_PROP_NAME});
      reader =
          new BufferedReader(
              new InputStreamReader(process.getInputStream(), Charset.forName("UTF-8")));

      String lastLine = "";
      String line;
      while ((line = reader.readLine()) != null) {
        lastLine = line;
      }
      metroHostPropValue = lastLine;
    } catch (Exception e) {
      FLog.w(TAG, "Failed to query for metro.host prop:", e);
      metroHostPropValue = "";
    } finally {
      try {
        if (reader != null) {
          reader.close();
        }
      } catch (Exception exc) {
      }
      if (process != null) {
        process.destroy();
      }
    }
    return metroHostPropValue;
  }

  // Cached value of the "metro.host" system property
  private static String metroHostPropValue = null;

  /**
   * Returns the IP address of the development server, depending on the environment.
   *
   * @param port the port number of the server
   * @return the IP address of the server
   */
  private static String getServerIpAddress(int port) {
    String ipAddress;
    String metroHostProp = getMetroHostPropValue();
    if (!metroHostProp.equals("")) {
      ipAddress = metroHostProp;
    } else if (isRunningOnGenymotion()) {
      ipAddress = GENYMOTION_LOCALHOST;
    } else if (isRunningOnStockEmulator()) {
      ipAddress = EMULATOR_LOCALHOST;
    } else {
      ipAddress = DEVICE_LOCALHOST;
    }
    return String.format(Locale.US, "%s:%d", ipAddress, port);
  }

  /**
   * Returns the integer value of a resource.
   *
   * @param context the app context
   * @param resId the resource ID
   * @return the integer value of the resource
   */
  private static int getIntegerResource(Context context, int resId) {
    Resources resources = context.getResources();
    return resources.getInteger(resId);
  }

}