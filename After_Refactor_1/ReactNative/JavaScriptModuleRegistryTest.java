/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.bridge;

import org.junit.Assert;
import org.junit.Test;

/** 
 * Tests for {@link JavaScriptModuleRegistry} 
 * 
 * Improvements made:
 * - Renamed test methods to better reflect their purpose
 * - Extracted constant strings for class names
 * - Added descriptive comments
 */
public class JavaScriptModuleRegistryTest {

  private interface TestJavaScriptModule extends JavaScriptModule {
    void doSomething();
  }

  private interface NestedInnerClass extends JavaScriptModule {
    void doSomething();
  }

  private static final String TEST_MODULE_NAME = "TestJavaScriptModule";
  private static final String NESTED_INNER_CLASS_NAME = "NestedInnerClass";

  /**
   * Tests that the correct JS module name is returned for a simple JavaScriptModule interface
   */
  @Test
  public void testGetJSModuleName_simpleModule() {
    String name = JavaScriptModuleRegistry.getJSModuleName(TestJavaScriptModule.class);
    Assert.assertEquals(TEST_MODULE_NAME, name);
  }

  /**
   * Tests that the correct JS module name is returned for a nested inner class of a Java class
   */
  @Test
  public void testGetJSModuleName_nestedInnerClass() {
    String name = JavaScriptModuleRegistry.getJSModuleName(NestedInnerClass.class);
    Assert.assertEquals(NESTED_INNER_CLASS_NAME, name);
  }
}

