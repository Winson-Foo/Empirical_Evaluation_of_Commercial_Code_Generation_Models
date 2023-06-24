package com.facebook.react.tests;

import com.facebook.react.bridge.JavaScriptModule;
import com.facebook.react.testing.AssertModule;
import com.facebook.react.testing.ReactAppInstrumentationTestCase;
import com.facebook.react.testing.ReactInstanceSpecForTest;
import com.facebook.react.uimanager.UIManagerModule;

/**
 * Tests for measuring view dimensions and layouts with {@link UIManagerModule}.
 */
public class MeasureLayoutTest extends ReactAppInstrumentationTestCase {

  /**
   * JavaScript module for making assertions in the test environment.
   */
  private interface MeasureLayoutTestJSModule extends JavaScriptModule {
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

  private MeasureLayoutTestJSModule mTestJSModule;
  private AssertModule mAssertModule;

  @Before
  public void setUpMeasureLayoutTest() throws Exception {
    mTestJSModule = getReactContext().getJSModule(MeasureLayoutTestJSModule.class);
    createAssertModule();
  }

  @After
  public void tearDownMeasureLayoutTest() {
    waitForBridgeAndUIIdle();
    mAssertModule.verifyAssertsAndReset();
  }

  @Override
  protected String getReactApplicationKeyUnderTest() {
    return "MeasureLayoutTestApp";
  }

  @Override
  protected ReactInstanceSpecForTest createReactInstanceSpecForTest() {
    return super.createReactInstanceSpecForTest().addNativeModule(mAssertModule);
  }

  /**
   * Tests {@link UIManagerModule#measure} on view A and view C.
   */
  @Test
  public void testMeasure() {
    mTestJSModule.verifyMeasureOnViewA();
    mTestJSModule.verifyMeasureOnViewC();
  }

  /**
   * Tests {@link UIManagerModule#measureLayout} on view C relative to view A, view B, and itself.
   */
  @Test
  public void testMeasureLayout() {
    mTestJSModule.verifyMeasureLayoutCRelativeToA();
    mTestJSModule.verifyMeasureLayoutCRelativeToB();
    mTestJSModule.verifyMeasureLayoutCRelativeToSelf();
  }

  /**
   * Tests {@link UIManagerModule#measureLayoutRelativeToParent} on view A, view B, and view C.
   */
  @Test
  public void testMeasureLayoutRelativeToParent() {
    mTestJSModule.verifyMeasureLayoutRelativeToParentOnViewA();
    mTestJSModule.verifyMeasureLayoutRelativeToParentOnViewB();
    mTestJSModule.verifyMeasureLayoutRelativeToParentOnViewC();
  }

  /**
   * Tests that an error callback is called when trying to measure a view relative to a non-existent ancestor.
   */
  @Test
  public void testMeasureLayoutCallsErrorCallbackWhenAncestorDoesNotExist() {
    mTestJSModule.verifyMeasureLayoutNonExistentAncestor();
  }

  /**
   * Tests that an error callback is called when trying to measure a non-existent view.
   */
  @Test
  public void testMeasureLayoutCallsErrorCallbackWhenViewDoesNotExist() {
    mTestJSModule.verifyMeasureLayoutNonExistentTag();
  }

  /**
   * Tests that an error callback is called when trying to measure a non-existent view relative to its parent.
   */
  @Test
  public void testMeasureLayoutRelativeToParentCallsErrorCallbackWhenViewDoesNotExist() {
    mTestJSModule.verifyMeasureLayoutRelativeToParentNonExistentTag();
  }

  /**
   * Tests that an error callback is called when trying to measure a view relative to a non-existent ancestor.
   */
  @Test
  public void testMeasureLayoutCallsErrorCallbackWhenViewIsNotChildOfAncestor() {
    mTestJSModule.verifyMeasureLayoutDRelativeToB();
  }

  /**
   * Creates the {@link AssertModule} instance used for making assertions in the test environment.
   */
  private void createAssertModule() {
    mAssertModule = new AssertModule();
  }
}