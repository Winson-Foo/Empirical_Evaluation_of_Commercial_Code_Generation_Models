/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.fabric.mounting.mountitems;

import androidx.annotation.NonNull;
import java.util.HashMap;
import java.util.Map;

/**
 * Utility class for Fabric components, this will be removed
 *
 * <p>TODO T97384889: remove this class when the component names are unified between JS - Android -
 * iOS - C++
 */
class FabricNameComponentMapping {

  private static @NonNull final Map<String, String> sComponentNames = new HashMap<>();

  static {
    // Load mapping from configuration file
    ConfigLoader.loadConfig(sComponentNames);
  }

  /** @return the name of component in the Fabric environment */
  static String getFabricComponentName(String componentName) {
    String component = sComponentNames.get(componentName);
    return component != null ? component : componentName;
  }

  static class ConfigLoader {
    // Path to configuration file containing mapping of original component names to Fabric component names
    private static final String CONFIG_FILE_PATH = "fabric_component_mapping.conf";

    static void loadConfig(Map<String, String> componentNames) {
      try {
        Properties props = new Properties();
        InputStream inputStream = FabricNameComponentMapping.class.getClassLoader().getResourceAsStream(CONFIG_FILE_PATH);
        props.load(inputStream);
        for (String key : props.stringPropertyNames()) {
          componentNames.put(key, props.getProperty(key));
        }
      } catch (IOException e) {
        // Log error if configuration file cannot be loaded
        Log.e("FabricNameComponentMapping", "Error loading configuration file: " + e.getMessage());
      }
    }
  }
}

