package com.facebook.react.modules.systeminfo;

import android.content.Context;
import android.content.res.Resources;
import android.os.Build;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.Locale;

public class AndroidInfoHelpers {

    private static final Logger LOGGER = LoggerFactory.getLogger(AndroidInfoHelpers.class);
    private static final String TAG = AndroidInfoHelpers.class.getSimpleName();

    private static final String EMULATOR_LOCALHOST = "10.0.2.2";
    private static final String GENYMOTION_LOCALHOST = "10.0.3.2";
    private static final String DEVICE_LOCALHOST = "localhost";
    private static final String METRO_HOST_PROP_NAME = "metro.host";

    private static String metroHostPropertyValue = null;

    /**
     * Checks if the app is running on a Genymotion device.
     *
     * @return true if running on Genymotion, false otherwise
     */
    private static boolean isRunningOnGenymotion() {
        return Build.FINGERPRINT.contains("vbox");
    }

    /**
     * Checks if the app is running on a stock emulator.
     *
     * @return true if running on stock emulator, false otherwise
     */
    private static boolean isRunningOnStockEmulator() {
        return Build.FINGERPRINT.contains("generic")
                || Build.FINGERPRINT.startsWith("google/sdk_gphone");
    }

    /**
     * Gets the server host IP address.
     *
     * @param port the port number
     * @return the server IP address
     */
    public static String getServerHost(int port) {
        return getServerIpAddress(port);
    }

    /**
     * Gets the server IP address using the dev server port specified in resources.
     *
     * @param context the Android context
     * @return the server IP address
     */
    public static String getServerHost(Context context) {
        return getServerIpAddress(getDevServerPort(context));
    }

    /**
     * Gets the adb reverse TCP command.
     *
     * @param port the port number
     * @return the adb reverse TCP command
     */
    public static String getAdbReverseTcpCommand(int port) {
        return "adb reverse tcp:" + port + " tcp:" + port;
    }

    /**
     * Gets the adb reverse TCP command using the dev server port specified in resources.
     *
     * @param context the Android context
     * @return the adb reverse TCP command
     */
    public static String getAdbReverseTcpCommand(Context context) {
        return getAdbReverseTcpCommand(getDevServerPort(context));
    }

    /**
     * Gets the inspector proxy host.
     *
     * @param context the Android context
     * @return the inspector proxy host IP address
     */
    public static String getInspectorProxyHost(Context context) {
        return getServerIpAddress(getInspectorProxyPort(context));
    }

    /**
     * Gets the friendly device name.
     *
     * @return the friendly device name
     */
    public static String getFriendlyDeviceName() {
        if (isRunningOnGenymotion()) {
            return Build.MODEL;
        } else {
            return Build.MODEL + " - " + Build.VERSION.RELEASE + " - API " + Build.VERSION.SDK_INT;
        }
    }

    /**
     * Gets the dev server port specified in resources.
     *
     * @param context the Android context
     * @return the dev server port
     */
    private static int getDevServerPort(Context context) {
        Resources resources = context.getResources();
        return resources.getInteger(R.integer.react_native_dev_server_port);
    }

    /**
     * Gets the inspector proxy port specified in resources.
     *
     * @param context the Android context
     * @return the inspector proxy port
     */
    private static int getInspectorProxyPort(Context context) {
        Resources resources = context.getResources();
        return resources.getInteger(R.integer.react_native_dev_server_port);
    }

    /**
     * Gets the server IP address based on the device type and metro host property.
     *
     * @param port the port number
     * @return the server IP address
     */
    private static String getServerIpAddress(int port) {
        String ipAddress;
        String metroHostProp = getMetroHostPropertyValue();
        if (!metroHostProp.isEmpty()) {
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
     * Gets the value of the metro host property.
     *
     * @return the value of the metro host property
     */
    private static synchronized String getMetroHostPropertyValue() {
        if (metroHostPropertyValue != null) {
            return metroHostPropertyValue;
        }

        Process process = null;
        BufferedReader reader = null;
        try {
            process = Runtime.getRuntime().exec(new String[]{"/system/bin/getprop", METRO_HOST_PROP_NAME});
            reader = new BufferedReader(new InputStreamReader(process.getInputStream(), Charset.forName("UTF-8")));

            String lastLine = "";
            String line;
            while ((line = reader.readLine()) != null) {
                lastLine = line;
            }
            metroHostPropertyValue = lastLine;
        } catch (Exception e) {
            LOGGER.warn("Failed to query for metro.host prop:", e);
            metroHostPropertyValue = "";
        } finally {
            try {
                if (reader != null) reader.close();
            } catch (Exception ignored) {}
            if (process != null) process.destroy();
        }
        return metroHostPropertyValue;
    }
}