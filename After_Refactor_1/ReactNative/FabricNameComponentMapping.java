/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.fabric.mounting.mountitems;

import java.util.Map;

/**
 * Utility class for mapping component names between JS - Android - iOS - C++ in Fabric components.
 */
public class FabricComponentNameMapper {

  private static final Map<String, String> COMPONENT_NAMES = Map.of(
          "View", "RCTView",
          "Image", "RCTImageView",
          "ScrollView", "RCTScrollView",
          "Slider", "RCTSlider",
          "ModalHostView", "RCTModalHostView",
          "Paragraph", "RCTText",
          "Text", "RCTText",
          "RawText", "RCTRawText",
          "ActivityIndicatorView", "AndroidProgressBar",
          "ShimmeringView", "RKShimmeringView",
          "TemplateView", "RCTTemplateView",
          "AxialGradientView", "RCTAxialGradientView",
          "Video", "RCTVideo",
          "Map", "RCTMap",
          "WebView", "RCTWebView",
          "Keyframes", "RCTKeyframes",
          "ImpressionTrackingView", "RCTImpressionTrackingView"
  );

  /**
   * @return the name of the Fabric component given its name in JS/Android/iOS/C++
   */
  public static String getFabricComponentName(String componentName) {
    return COMPONENT_NAMES.getOrDefault(componentName, componentName);
  }
}