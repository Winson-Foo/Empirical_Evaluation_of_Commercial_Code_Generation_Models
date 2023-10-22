package com.facebook.react.modules.appregistry;

import com.facebook.react.bridge.JavaScriptModule;
import com.facebook.react.bridge.WritableMap;

/**
 * The AppRegistry module interface provides methods for launching and managing React Native
 * applications.
 */
public interface AppRegistryModule extends JavaScriptModule {

  /**
   * Launches a React Native application with the specified key and parameters.
   *
   * @param appKey         The unique key of the application to launch.
   * @param launchOptions  The launch options for the application.
   */
  void launchApplication(String appKey, WritableMap launchOptions);

  /**
   * Unmounts the React component at the specified root node tag.
   *
   * @param rootNodeTag    The tag of the root node of the component to unmount.
   */
  void unmountComponentAtRootTag(int rootNodeTag);

  /**
   * Starts a headless task with the specified task ID, key, and data.
   *
   * @param taskId         The unique ID of the task to start.
   * @param taskKey        The key of the task to start.
   * @param taskData       The data to pass to the task.
   */
  void startHeadlessTask(long taskId, String taskKey, WritableMap taskData);
  
  /**
   * The unique key used to launch the React Native Fabric application.
   */
  String FABRIC_APP_KEY = "ReactNativeFabric";

  /**
   * The unique key used to launch the default React Native application.
   */
  String DEFAULT_APP_KEY = "main";
  
  /**
   * The key for passing the component class name to the React Native application launch options.
   */
  String COMPONENT_CLASS_NAME_KEY = "componentName";
  
  /**
   * The key for passing the initial props to the React Native application launch options.
   */
  String INITIAL_PROPS_KEY = "initialProps";
  
} 