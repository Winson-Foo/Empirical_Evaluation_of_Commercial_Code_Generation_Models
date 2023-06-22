/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.modules.appregistry;

import com.facebook.react.bridge.JavaScriptModule;

/**
 * Interface for launching React applications.
 */
public interface ReactAppLauncher extends JavaScriptModule {

  /**
   * Launches a React application with the given name and parameters.
   *
   * @param appName The name of the application to launch.
   * @param appParams The parameters to pass to the application.
   */
  void launchReactApp(String appName, AppParams appParams);

  /**
   * Unmounts a React component at the given root tag.
   *
   * @param rootNodeTag The root tag of the component to unmount.
   */
  void unmountReactComponentAtRootTag(int rootNodeTag);

  /**
   * Starts a background task with the given ID, key, and data.
   *
   * @param taskId The ID of the task to start.
   * @param taskKey The key of the task to start.
   * @param data The data to pass to the task.
   */
  void startBackgroundTask(int taskId, String taskKey, AppParams data);
}

/**
 * Parameters to pass to a React application.
 */
class AppParams {
  String appName;
  Object appParams;
  // Add more properties as needed.
}