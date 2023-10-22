/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.tests;

import com.facebook.react.bridge.JavaScriptModule;
import com.facebook.react.testing.ReactAppInstrumentationTestCase;
import com.facebook.react.testing.ReactInstanceSpecForTest;
import com.facebook.react.testing.AssertModule;
import com.facebook.react.uimanager.UIManagerModule;

/**
 * Tests for measuring dimensions and layouts of views in a hierarchy
 */
public class MeasureLayoutTest extends ReactAppInstrumentationTestCase {

  private interface MeasureLayoutTestModule extends JavaScriptModule {
    public void verifyMeasureOnViewA();

    public void verifyMeasureOnViewC();

    public void verifyMeasureLayoutCRelativeToA();

    public void verifyMeasureLayoutCRelativeToB();

    public void verifyMeasureLayoutCRelativeToSelf();

    public void verifyMeasureLayoutRelativeToParentOnViewA();

    public void verifyMeasureLayoutRelativeToParentOnViewB();

    public void verifyMeasureLayoutRelativeToParentOnViewC();

    public void verifyMeasureLayoutDRelativeToB();

    public void verifyMeasureLayoutNonExistentTag();

    public void verifyMeasureLayoutNonExistentAncestor();

    public void verifyMeasureLayoutRelativeToParentNonExistentTag();
  }

  private MeasureLayoutTestModule testJSModule;
  private AssertModule assertModule;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    testJSModule = getReactContext().getJSModule(MeasureLayoutTestModule.class);
  }

  @Override
  protected String getReactApplicationKeyUnderTest() {
    return "MeasureLayoutTestApp";
  }

  @Override
  protected ReactInstanceSpecForTest createReactInstanceSpecForTest() {
    assertModule = new AssertModule();
    return super.createReactInstanceSpecForTest().addNativeModule(assertModule);
  }

  /**
   * Measure the dimensions of views A and C
   */
  public void testMeasureDimensions() {
    testJSModule.verifyMeasureOnViewA();
    waitForBridgeIdleAndVerifyAsserts();
    testJSModule.verifyMeasureOnViewC();
    waitForBridgeIdleAndVerifyAsserts();
  }

  /**
   * Test measure layout calls between views
   */
  public void testMeasureLayout() {
    testJSModule.verifyMeasureLayoutCRelativeToA();
    waitForBridgeIdleAndVerifyAsserts();
    testJSModule.verifyMeasureLayoutCRelativeToB();
    waitForBridgeIdleAndVerifyAsserts();
    testJSModule.verifyMeasureLayoutCRelativeToSelf();
    waitForBridgeIdleAndVerifyAsserts();
  }

  /**
   * Test measure layout calls relative to parent views
   */
  public void testMeasureLayoutRelativeToParent() {
    testJSModule.verifyMeasureLayoutRelativeToParentOnViewA();
    waitForBridgeIdleAndVerifyAsserts();
    testJSModule.verifyMeasureLayoutRelativeToParentOnViewB();
    waitForBridgeIdleAndVerifyAsserts();
    testJSModule.verifyMeasureLayoutRelativeToParentOnViewC();
    waitForBridgeIdleAndVerifyAsserts();
  }

  /**
   * Test measure layout calls where the view is not child of an ancestor
   */
  public void testMeasureLayoutNotChildOfAncestor() {
    testJSModule.verifyMeasureLayoutDRelativeToB();
    waitForBridgeIdleAndVerifyAsserts();
  }

  /**
   * Test measure layout calls when the view does not exist
   */
  public void testMeasureLayoutDoesNotExist() {
    testJSModule.verifyMeasureLayoutNonExistentTag();
    waitForBridgeIdleAndVerifyAsserts();
  }

  /**
   * Test measure layout calls when ancestor does not exist
   */
  public void testMeasureLayoutAncestorDoesNotExist() {
    testJSModule.verifyMeasureLayoutNonExistentAncestor();
    waitForBridgeIdleAndVerifyAsserts();
  }

  /**
   * Test measure layout calls relative to parent views when the view does not exist
   */
  public void testMeasureLayoutRelativeToParentDoesNotExist() {
    testJSModule.verifyMeasureLayoutRelativeToParentNonExistentTag();
    waitForBridgeIdleAndVerifyAsserts();
  }

  /**
  * Wait for the bridge to be idle and verify the assertions 
  */
  private void waitForBridgeIdleAndVerifyAsserts() {
    waitForBridgeAndUIIdle();
    assertModule.verifyAssertsAndReset();
  }
}

