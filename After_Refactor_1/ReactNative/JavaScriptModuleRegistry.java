package com.facebook.react.bridge;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

public final class JavaScriptModuleRegistry {
  private final HashMap<Class<? extends JavaScriptModule>, JavaScriptModule> mModuleInstances;

  public JavaScriptModuleRegistry() {
    mModuleInstances = new HashMap<>();
  }

  public synchronized <T extends JavaScriptModule> T getJavaScriptModule(
      Class<T> moduleInterface) {
    JavaScriptModule module = mModuleInstances.get(moduleInterface);
    if (module != null) {
      return (T) module;
    }

    JavaScriptModule interfaceProxy =
        (JavaScriptModule)
            Proxy.newProxyInstance(
                moduleInterface.getClassLoader(),
                new Class[] {moduleInterface},
                new JavaScriptModuleInvocationHandler(moduleInterface));
    mModuleInstances.put(moduleInterface, interfaceProxy);
    return (T) interfaceProxy;
  }

  public static String getJSModuleName(Class<? extends JavaScriptModule> jsModuleInterface) {
    return jsModuleInterface.getSimpleName();
  }

  private static class JavaScriptModuleInvocationHandler implements InvocationHandler {
    private final Class<? extends JavaScriptModule> mModuleInterface;

    public JavaScriptModuleInvocationHandler(
        Class<? extends JavaScriptModule> moduleInterface) {
      mModuleInterface = moduleInterface;

      if (ReactBuildConfig.DEBUG) {
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
    }

    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
      CatalystInstance instance = ReactContextBaseJavaModule.getCatalystInstance();
      if (instance == null) {
        throw new IllegalStateException("CatalystInstance not available");
      }
      NativeArray jsArgs = args != null ? Arguments.fromJavaArgs(args) : new WritableNativeArray();
      instance.callFunction(getJSModuleName(mModuleInterface), method.getName(), jsArgs);
      return null;
    }
  }
}