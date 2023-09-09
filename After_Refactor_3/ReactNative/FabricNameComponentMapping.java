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

  private static @NonNull final Map<String, String> COMPONENT_NAMES = new HashMap<>();

  static {
    setComponentNames();
  }

  private FabricNameComponentMapping() {}

  private static void setComponentNames() {
  // TODO T97384889: unify component names between JS - Android - iOS - C++
    addToComponentNames("View", "RCTView");
    addToComponentNames("Image", "RCTImageView");
    addToComponentNames("ScrollView", "RCTScrollView");
    addToComponentNames("Slider", "RCTSlider");
    addToComponentNames("ModalHostView", "RCTModalHostView");
    addToComponentNames("Paragraph", "RCTText");
    addToComponentNames("Text", "RCText");
    addToComponentNames("RawText", "RCTRawText");
    addToComponentNames("ActivityIndicatorView", "AndroidProgressBar");
    addToComponentNames("ShimmeringView", "RKShimmeringView");
    addToComponentNames("TemplateView", "RCTTemplateView");
    addToComponentNames("AxialGradientView", "RCTAxialGradientView");
    addToComponentNames("Video", "RCTVideo");
    addToComponentNames("Map", "RCTMap");
    addToComponentNames("WebView", "RCTWebView");
    addToComponentNames("Keyframes", "RCTKeyframes");
    addToComponentNames("ImpressionTrackingView", "RCTImpressionTrackingView");
  }

  private static void addToComponentNames(String key, String value) {
    COMPONENT_NAMES.put(key, value);
  }

  /** @return the name of the component in the Fabric environment */
  static String getFabricComponentName(String componentName) {
    String component = COMPONENT_NAMES.get(componentName);
    return component != null ? component : componentName;
  }
}