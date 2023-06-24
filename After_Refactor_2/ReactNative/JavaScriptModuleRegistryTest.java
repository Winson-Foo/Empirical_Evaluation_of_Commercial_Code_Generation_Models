// JavaScriptModuleRegistryTest.java

package com.facebook.react.bridge;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class JavaScriptModuleRegistryTest {

  @Test
  public void shouldReturnModuleNameForSingleInterface() {
    String name = JavaScriptModuleRegistry.getJSModuleName(TestJavaScriptModule.class);
    assertEquals("TestJavaScriptModule", name);
  }

  @Test
  public void shouldStripOuterClassFromModuleName() {
    String name = JavaScriptModuleRegistry.getJSModuleName(NestedInnerClass.class);
    assertEquals("NestedInnerClass", name);
  }

  private interface TestJavaScriptModule extends JavaScriptModule {
    void doSomething();
  }

  private interface NestedInnerClass extends JavaScriptModule {
    void doSomething();
  }
}

// TestJavaScriptModule.java

package com.facebook.react.bridge;

public interface TestJavaScriptModule extends JavaScriptModule {
  void doSomething();
}

// NestedInnerClass.java

package com.facebook.react.bridge;

public interface NestedInnerClass extends JavaScriptModule {
  void doSomething();
}

