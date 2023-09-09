/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.bridge;

import java.util.List;

/**
 * Interface for providing JSI modules to the React Native bridge.
 */
public interface JSIModuleProvider {
  
  /**
   * Returns a list of JSI modules to be registered with the bridge.
   *
   * @param reactApplicationContext the context of the React Native application.
   * @param jsContext the holder for the JavaScript runtime context.
   * @return the list of JSI module specifications.
   */
  List<JSIModuleSpec> getJSIModules(
      ReactApplicationContext reactApplicationContext, JavaScriptContextHolder jsContext);

  /**
   * Returns an empty list of JSI modules.
   * This method is provided as a default implementation to avoid breaking changes.
   *
   * @param reactApplicationContext the context of the React Native application.
   * @param jsContext the holder for the JavaScript runtime context.
   * @return an empty list of JSI module specifications.
   */
  default List<JSIModuleSpec> getEmptyJSIModules(
      ReactApplicationContext reactApplicationContext, JavaScriptContextHolder jsContext) {
    return List.of();
  }
}