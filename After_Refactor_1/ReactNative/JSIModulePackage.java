// File: JSIModulePackage.java

package com.facebook.react.bridge;

import com.google.common.collect.ImmutableList;

/**
 * Interface used to initialize JSI modules into the JSI bridge.
 */
public interface JSIModulePackage {

  /**
   * Gets the list of JSI modules provided by this package.
   *
   * @param reactApplicationContext The react application context.
   * @param jsContext The JavaScript context holder.
   * @return The immutable list of JSI module providers.
   */
  ImmutableList<JSIModuleProvider> getJSIModuleProviders(
      ReactApplicationContext reactApplicationContext, final JavaScriptContextHolder jsContext);
}

