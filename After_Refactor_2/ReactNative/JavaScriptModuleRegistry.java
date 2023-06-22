package com.facebook.react.bridge;

import androidx.annotation.Nullable;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Class responsible for holding all the {@link JavaScriptModule}s. Uses Java proxy objects to
 * dispatch method calls on JavaScriptModules to the bridge using the corresponding module and
 * method ids so the proper function is executed in JavaScript.
 */
public final class JavaScriptModuleRegistry {

    private final Map<Class<? extends JavaScriptModule>, JavaScriptModule> moduleInstances;

    public JavaScriptModuleRegistry() {
        moduleInstances = new HashMap<>();
    }

    /**
     * Returns a JavaScriptModule implementation for the given module interface.
     * If such an implementation is already cached, returns it from the cache.
     * Otherwise, creates a new proxy implementation and caches it for future use.
     * @param catalystInstance The catalyst instance to which the JavaScript module calls should be dispatched to.
     * @param moduleInterface The Java interface corresponding to the JavaScript module.
     * @return An instance of the given JavaScript module interface.
     */
    public synchronized <T extends JavaScriptModule> T getJavaScriptModule(
            CatalystInstance catalystInstance, Class<T> moduleInterface) {
        JavaScriptModule module = moduleInstances.get(moduleInterface);
        if (module != null) {
            return (T) module;
        }

        JavaScriptModule interfaceProxy = createJavaScriptModuleProxy(catalystInstance, moduleInterface);
        moduleInstances.put(moduleInterface, interfaceProxy);
        return (T) interfaceProxy;
    }

    private static JavaScriptModule createJavaScriptModuleProxy(
            CatalystInstance catalystInstance, Class<? extends JavaScriptModule> moduleInterface) {
        if (ReactBuildConfig.DEBUG) {
            checkMethodOverloading(moduleInterface);
        }

        String jsModuleName = getJSModuleName(moduleInterface);
        InvocationHandler invocationHandler = new JavaScriptModuleInvocationHandler(
                catalystInstance, moduleInterface, jsModuleName);

        return (JavaScriptModule) Proxy.newProxyInstance(
                moduleInterface.getClassLoader(), new Class[]{moduleInterface}, invocationHandler);
    }

    private static void checkMethodOverloading(Class<? extends JavaScriptModule> moduleInterface) {
        Set<String> methodNames = new HashSet<>();
        for (Method method : moduleInterface.getDeclaredMethods()) {
            if (!methodNames.add(method.getName())) {
                throw new AssertionError(
                        "Method overloading is unsupported: "
                                + moduleInterface.getName()
                                + "#"
                                + method.getName());
            }
        }
    }

    /**
     * Returns the JavaScript module name for a given Java interface.
     * @param jsModuleInterface The Java interface corresponding to the JavaScript module.
     * @return The JavaScript module name for the interface.
     */
    public static String getJSModuleName(Class<? extends JavaScriptModule> jsModuleInterface) {
        String className = jsModuleInterface.getSimpleName();
        int dollarSignIndex = className.lastIndexOf('$');
        if (dollarSignIndex != -1) {
            className = className.substring(dollarSignIndex + 1);
        }
        return className;
    }

    private static class JavaScriptModuleInvocationHandler implements InvocationHandler {
        private final CatalystInstance catalystInstance;
        private final Class<? extends JavaScriptModule> moduleInterface;
        private final String jsModuleName;

        public JavaScriptModuleInvocationHandler(
                CatalystInstance catalystInstance, Class<? extends JavaScriptModule> moduleInterface,
                String jsModuleName) {
            this.catalystInstance = catalystInstance;
            this.moduleInterface = moduleInterface;
            this.jsModuleName = jsModuleName;
        }

        @Override
        public @Nullable Object invoke(Object proxy, Method method, @Nullable Object[] args) throws Throwable {
            NativeArray jsArgs = args != null ? Arguments.fromJavaArgs(args) : new WritableNativeArray();
            catalystInstance.callFunction(jsModuleName, method.getName(), jsArgs);
            return null;
        }
    }
}