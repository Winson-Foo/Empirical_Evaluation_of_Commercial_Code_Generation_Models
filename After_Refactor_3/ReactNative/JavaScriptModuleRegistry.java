package com.facebook.react.bridge;

import androidx.annotation.Nullable;
import com.facebook.react.common.build.ReactBuildConfig;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

/**
 * Class responsible for holding all the {@link JavaScriptModule}s. Uses Java proxy objects to
 * dispatch method calls on JavaScriptModules to the bridge using the corresponding module and
 * method ids so the proper function is executed in JavaScript.
 */
public final class JavaScriptModuleRegistry {
  private final HashMap<Class<? extends JavaScriptModule>, JavaScriptModule> mModuleInstances;

  public JavaScriptModuleRegistry() {
    mModuleInstances = new HashMap<>();
  }

  /**
   * Returns the JavaScript module corresponding to the given module class interface.
   *
   * @param catalystInstance the catalyst instance to use
   * @param moduleInterface the class interface of the JavaScript module
   * @return the JavaScript module
   */
  public synchronized <T extends JavaScriptModule> T getJavaScriptModule(
      CatalystInstance catalystInstance, Class<T> moduleInterface) {
    JavaScriptModule module = mModuleInstances.get(moduleInterface);
    if (module != null) {
      return (T) module;
    }

    T interfaceProxy = createJavaScriptModuleProxy(catalystInstance, moduleInterface);
    mModuleInstances.put(moduleInterface, interfaceProxy);
    return interfaceProxy;
  }

  private <T extends JavaScriptModule> T createJavaScriptModuleProxy(
      CatalystInstance catalystInstance, Class<T> moduleInterface) {
    InvocationHandler invocationHandler =
        new JavaScriptModuleInvocationHandler(catalystInstance, moduleInterface);
    return (T)
        Proxy.newProxyInstance(moduleInterface.getClassLoader(), new Class[] {moduleInterface},
            invocationHandler);
  }

  private static class JavaScriptModuleInvocationHandler implements InvocationHandler {
    private final CatalystInstance mCatalystInstance;
    private final Class<? extends JavaScriptModule> mModuleInterface;
    private @Nullable String mJSModuleName;

    public JavaScriptModuleInvocationHandler(
        CatalystInstance catalystInstance, Class<? extends JavaScriptModule> moduleInterface) {
      mCatalystInstance = catalystInstance;
      mModuleInterface = moduleInterface;

      // Check for method overloading in debug mode
      if (ReactBuildConfig.DEBUG) {
        checkForMethodOverloading();
      }
    }

    private void checkForMethodOverloading() {
      Set<String> methodNames = new HashSet<>();
      for (Method method : mModuleInterface.getDeclaredMethods()) {
        if (!methodNames.add(method.getName())) {
          throw new AssertionError(
              "Method overloading is unsupported: "
                  + mModuleInterface.getName()
                  + "#"
                  + method.getName());
        }
      }
    }

    private String getJSModuleName() {
      if (mJSModuleName == null) {
        mJSModuleName = JavaScriptModuleUtils.getJSModuleName(mModuleInterface);
      }
      return mJSModuleName;
    }

    @Override
    public @Nullable Object invoke(Object proxy, Method method, @Nullable Object[] args)
        throws Throwable {
      NativeArray jsArgs = args != null ? Arguments.fromJavaArgs(args) : new WritableNativeArray();
      mCatalystInstance.callFunction(getJSModuleName(), method.getName(), jsArgs);
      return null;
    }
  }
}

// Utility class for JavaScript module related functions
final class JavaScriptModuleUtils {
  /**
   * Returns the JavaScript module name for the given module class interface.
   *
   * @param jsModuleInterface the class interface of the JavaScript module
   * @return the JavaScript module name
   */
  public static String getJSModuleName(Class<? extends JavaScriptModule> jsModuleInterface) {
    // With proguard obfuscation turned on, proguard apparently (poorly) emulates inner
    // classes or something because Class#getSimpleName() no longer strips the outer
    // class name. We manually strip it here if necessary.
    String name = jsModuleInterface.getSimpleName();
    int dollarSignIndex = name.lastIndexOf('$');
    if (dollarSignIndex != -1) {
      name = name.substring(dollarSignIndex + 1);
    }
    return name;
  }
}

