// CatalystMeasureLayoutTestModule.java
package com.facebook.react.tests;

import com.facebook.react.bridge.JavaScriptModule;

public interface CatalystMeasureLayoutTestModule extends JavaScriptModule {
  void verifyMeasureOnViewA();

  void verifyMeasureOnViewC();

  void verifyMeasureLayoutCRelativeToA();

  void verifyMeasureLayoutCRelativeToB();

  void verifyMeasureLayoutCRelativeToSelf();

  void verifyMeasureLayoutRelativeToParentOnViewA();

  void verifyMeasureLayoutRelativeToParentOnViewB();

  void verifyMeasureLayoutRelativeToParentOnViewC();

  void verifyMeasureLayoutDRelativeToB();

  void verifyMeasureLayoutNonExistentTag();

  void verifyMeasureLayoutNonExistentAncestor();

  void verifyMeasureLayoutRelativeToParentNonExistentTag();
}

// UIManagerMeasureLayoutTest.java
package com.facebook.react.tests;

import com.facebook.react.testing.AssertModule;
import com.facebook.react.testing.ReactAppInstrumentationTestCase;
import com.facebook.react.testing.ReactInstanceSpecForTest;
import com.facebook.react.uimanager.UIManagerModule;

public class UIManagerMeasureLayoutTest extends ReactAppInstrumentationTestCase {

  private static final int VIEW_A_TAG = 1;
  private static final int VIEW_B_TAG = 2;
  private static final int VIEW_C_TAG = 3;
  private static final int VIEW_D_TAG = 4;

  private static final int VIEW_A_WIDTH = 500;
  private static final int VIEW_A_HEIGHT = 500;
  private static final int VIEW_B_WIDTH = 200;
  private static final int VIEW_B_HEIGHT = 300;
  private static final int VIEW_B_LEFT = 50;
  private static final int VIEW_B_TOP = 80;
  private static final int VIEW_C_WIDTH = 50;
  private static final int VIEW_C_HEIGHT = 150;
  private static final int VIEW_C_LEFT = 150;
  private static final int VIEW_C_TOP = 150;
  private static final int VIEW_D_WIDTH = 50;
  private static final int VIEW_D_HEIGHT = 200;
  private static final int VIEW_D_LEFT = 400;
  private static final int VIEW_D_TOP = 100;

  private static final String MODULE_NAME = "CatalystMeasureLayoutTestModule";

  private CatalystMeasureLayoutTestModule mTestJSModule;
  private AssertModule mAssertModule;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    mTestJSModule = getReactContext().getJSModule(CatalystMeasureLayoutTestModule.class);
  }

  @Override
  protected String getReactApplicationKeyUnderTest() {
    return "MeasureLayoutTestApp";
  }

  @Override
  protected ReactInstanceSpecForTest createReactInstanceSpecForTest() {
    mAssertModule = new AssertModule();
    return super.createReactInstanceSpecForTest().addNativeModule(mAssertModule);
  }

  private void waitForBridgeIdleAndVerifyAsserts() {
    waitForBridgeAndUIIdle();
    mAssertModule.verifyAssertsAndReset();
  }

  public void testMeasureOnViews() {
    // Verify the measure function works correctly on views A and C
    mTestJSModule.verifyMeasureOnViewA();
    waitForBridgeIdleAndVerifyAsserts();
    mTestJSModule.verifyMeasureOnViewC();
    waitForBridgeIdleAndVerifyAsserts();
  }

  public void testMeasureLayoutRelativeToAncestor() {
    // Verify the measure layout function works correctly when measuring C relative to A and B
    mTestJSModule.verifyMeasureLayoutCRelativeToA();
    waitForBridgeIdleAndVerifyAsserts();
    mTestJSModule.verifyMeasureLayoutCRelativeToB();
    waitForBridgeIdleAndVerifyAsserts();
    mTestJSModule.verifyMeasureLayoutCRelativeToSelf();
    waitForBridgeIdleAndVerifyAsserts();
  }

  public void testMeasureLayoutRelativeToParent() {
    // Verify the measure layout function works correctly when measuring C and B relative to their parents
    mTestJSModule.verifyMeasureLayoutRelativeToParentOnViewA();
    waitForBridgeIdleAndVerifyAsserts();
    mTestJSModule.verifyMeasureLayoutRelativeToParentOnViewB();
    waitForBridgeIdleAndVerifyAsserts();
    mTestJSModule.verifyMeasureLayoutRelativeToParentOnViewC();
    waitForBridgeIdleAndVerifyAsserts();
  }

  public void testMeasureLayoutCallsErrorCallbackWhenViewIsNotChildOfAncestor() {
    // Verify that an error callback is called when measuring D relative to B
    mTestJSModule.verifyMeasureLayoutDRelativeToB();
    waitForBridgeIdleAndVerifyAsserts();
  }

  public void testMeasureLayoutCallsErrorCallbackWhenViewDoesNotExist() {
    // Verify that an error callback is called when measuring a non-existent view
    mTestJSModule.verifyMeasureLayoutNonExistentTag();
    waitForBridgeIdleAndVerifyAsserts();
  }

  public void testMeasureLayoutCallsErrorCallbackWhenAncestorDoesNotExist() {
    // Verify that an error callback is called when measuring a view relative to a non-existent ancestor
    mTestJSModule.verifyMeasureLayoutNonExistentAncestor();
    waitForBridgeIdleAndVerifyAsserts();
  }

  public void testMeasureLayoutRelativeToParentCallsErrorCallbackWhenViewDoesNotExist() {
    // Verify that an error callback is called when measuring a non-existent view relative to its parent
    mTestJSModule.verifyMeasureLayoutRelativeToParentNonExistentTag();
    waitForBridgeIdleAndVerifyAsserts();
  }
}