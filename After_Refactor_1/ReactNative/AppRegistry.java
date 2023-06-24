// AppRegistry.java
package com.facebook.react.modules.appregistry;

import com.facebook.react.bridge.JavaScriptModule;
import com.facebook.react.bridge.WritableMap;

/**
 * JS module interface - main entry point for launching React application for a given key.
 */
public interface AppRegistry extends JavaScriptModule {

  /**
   * Run the specified React Native application with the given key and parameters.
   * @param appKey the key for the app to run
   * @param appParams the parameters for the app to run
   */
  void runApplication(String appKey, WritableMap appParams);

  /**
   * Unmount the React Native application component at the specified root tag.
   * @param rootNodeTag the root tag of the component to unmount
   */
  void unmountApplicationComponentAtRootTag(int rootNodeTag);

  /**
   * Start a new headless task with the specified ID, key, and data.
   * @param taskId the ID of the new headless task
   * @param taskKey the key for the new headless task
   * @param data the data for the new headless task
   */
  void startHeadlessTask(int taskId, String taskKey, WritableMap data);
}