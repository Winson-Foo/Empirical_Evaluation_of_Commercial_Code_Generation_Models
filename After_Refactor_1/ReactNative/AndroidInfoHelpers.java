/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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

public class AndroidInfoHelpers {

  public static final String EMULATOR_LOCALHOST = "10.0.2.2";
  public static final String GENYMOTION_LOCALHOST = "10.0.3.2";
  public static final String DEVICE_LOCALHOST = "localhost";

  public static final String METRO_HOST_PROP_NAME = "metro.host";

  private static final String TAG = AndroidInfoHelpers.class.getSimpleName();

  private AndroidInfoHelpers() {}

  /**
   * Returns the IP address and port of the dev server.
   */
  public static String getServerHost(Context context) {
    int port = getDevServerPort(context);
    return getServerIpAddress(port);
  }

  /**
   * Returns the IP address and port of the dev server with the specified port.
   */
  public static String getServerHost(int port) {
    return getServerIpAddress(port);
  }

  /**
   * Returns the command for adb reverse tcp port.
   */
  public static String getAdbReverseTcpCommand(Context context) {
    int port = getDevServerPort(context);
    return "adb reverse tcp:" + port + " tcp:" + port;
  }

  /**
   * Returns the command for adb reverse tcp port with the specified port.
   */
  public static String getAdbReverseTcpCommand(int port) {
    return "adb reverse tcp:" + port + " tcp:" + port;
  }

  /**
   * Returns the IP address of the inspector proxy with the specified context.
   */
  public static String getInspectorProxyHost(Context context) {
    int port = getInspectorProxyPort(context);
    return getServerIpAddress(port);
  }

  /**
   * Returns the friendly device name.
   */
  public static String getFriendlyDeviceName() {
    if (isRunningOnGenymotion()) {
      return Build.MODEL;
    } else {
      return Build.MODEL + " - " + Build.VERSION.RELEASE + " - API " + Build.VERSION.SDK_INT;
    }
  }

  private static int getDevServerPort(Context context) {
    return getIntegerFromResources(context, R.integer.react_native_dev_server_port);
  }

  private static int getInspectorProxyPort(Context context) {
    return getIntegerFromResources(context, R.integer.react_native_dev_server_port);
  }

  private static int getIntegerFromResources(Context context, int resourceId) {
    Resources resources = context.getResources();
    return resources.getInteger(resourceId);
  }

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

  private static boolean isRunningOnGenymotion() {
    return Build.FINGERPRINT.contains("vbox");
  }

  private static boolean isRunningOnStockEmulator() {
    return Build.FINGERPRINT.contains("generic")
        || Build.FINGERPRINT.startsWith("google/sdk_gphone");
  }

  private static String metroHostPropValue = null;

  /**
   * Returns the value of the metro host property.
   */
  private static synchronized String getMetroHostPropValue() {
    if (metroHostPropValue != null) {
      return metroHostPropValue;
    }
    Process process = null;
    BufferedReader reader = null;
    try {
      process = Runtime.getRuntime().exec(new String[] {"/system/bin/getprop", METRO_HOST_PROP_NAME});
      reader = new BufferedReader(new InputStreamReader(process.getInputStream(), Charset.forName("UTF-8")));

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
}