/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.uimanager;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.reset;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;

import androidx.annotation.Nullable;
import com.facebook.react.bridge.JavaOnlyMap;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.uimanager.annotations.ReactProp;
import com.facebook.react.uimanager.annotations.ReactPropGroup;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mockito;
import org.mockito.junit.MockitoJUnitRunner;
import org.robolectric.RobolectricTestRunner;

/**
 * Test {@link ReactProp} annotation for {@link ReactShadowNode}.
 * More comprehensive test of this annotation can be found in {@link ReactPropAnnotationSetterTest}
 * where we test all possible types of properties to be updated.
 */
@RunWith(MockitoJUnitRunner.class)
public class ReactPropForShadowNodeSetterTest {

  public interface ViewManagerUpdatesReceiver {
    void onBooleanSetterCalled(boolean value);

    void onIntSetterCalled(int value);

    void onDoubleSetterCalled(double value);

    void onFloatSetterCalled(float value);

    void onStringSetterCalled(String value);

    void onBoxedBooleanSetterCalled(Boolean value);

    void onBoxedIntSetterCalled(Integer value);

    void onArraySetterCalled(ReadableArray value);

    void onMapSetterCalled(ReadableMap value);

    void onFloatGroupPropSetterCalled(int index, float value);

    void onIntGroupPropSetterCalled(int index, int value);

    void onBoxedIntGroupPropSetterCalled(int index, Integer value);
  }

  public static ReactStylesDiffMap buildStyles(Object... keysAndValues) {
    return new ReactStylesDiffMap(JavaOnlyMap.of(keysAndValues));
  }

  private class ShadowViewUnderTest extends ReactShadowNodeImpl {

    private ViewManagerUpdatesReceiver viewManagerUpdatesReceiver;

    private ShadowViewUnderTest(ViewManagerUpdatesReceiver viewManagerUpdatesReceiver) {
      this.viewManagerUpdatesReceiver = viewManagerUpdatesReceiver;
    }

    @ReactProp(name = "boolProp")
    public void setBoolProp(boolean value) {
      viewManagerUpdatesReceiver.onBooleanSetterCalled(value);
    }

    @ReactProp(name = "stringProp")
    public void setStringProp(@Nullable String value) {
      viewManagerUpdatesReceiver.onStringSetterCalled(value);
    }

    @ReactPropGroup(
        names = {
          "floatGroupPropFirst",
          "floatGroupPropSecond",
        })
    public void setFloatGroupProp(int index, float value) {
      viewManagerUpdatesReceiver.onFloatGroupPropSetterCalled(index, value);
    }
  }

  private ViewManagerUpdatesReceiver updatesReceiverMock;
  private ShadowViewUnderTest shadowView;

  @Before
  public void setup() {
    updatesReceiverMock = mock(ViewManagerUpdatesReceiver.class);
    shadowView = new ShadowViewUnderTest(updatesReceiverMock);
  }

  @Test
  public void testBooleanSetter() {
    shadowView.updateProperties(buildStyles("boolProp", true));
    verify(updatesReceiverMock).onBooleanSetterCalled(true);
    verifyNoMoreInteractions(updatesReceiverMock);
    reset(updatesReceiverMock);

    shadowView.updateProperties(buildStyles("boolProp", false));
    verify(updatesReceiverMock).onBooleanSetterCalled(false);
    verifyNoMoreInteractions(updatesReceiverMock);
    reset(updatesReceiverMock);

    shadowView.updateProperties(buildStyles("boolProp", null));
    verify(updatesReceiverMock).onBooleanSetterCalled(false);
    verifyNoMoreInteractions(updatesReceiverMock);
    reset(updatesReceiverMock);
  }

  @Test
  public void testStringSetter() {
    shadowView.updateProperties(buildStyles("stringProp", "someRandomString"));
    verify(updatesReceiverMock).onStringSetterCalled("someRandomString");
    verifyNoMoreInteractions(updatesReceiverMock);
    reset(updatesReceiverMock);

    shadowView.updateProperties(buildStyles("stringProp", null));
    verify(updatesReceiverMock).onStringSetterCalled(null);
    verifyNoMoreInteractions(updatesReceiverMock);
    reset(updatesReceiverMock);
  }

  @Test
  public void testFloatGroupSetter() {
    shadowView.updateProperties(buildStyles("floatGroupPropFirst", 11.0));
    verify(updatesReceiverMock).onFloatGroupPropSetterCalled(0, 11.0f);
    verifyNoMoreInteractions(updatesReceiverMock);
    reset(updatesReceiverMock);

    shadowView.updateProperties(buildStyles("floatGroupPropSecond", -111.0));
    verify(updatesReceiverMock).onFloatGroupPropSetterCalled(1, -111.0f);
    verifyNoMoreInteractions(updatesReceiverMock);
    reset(updatesReceiverMock);

    shadowView.updateProperties(buildStyles("floatGroupPropSecond", null));
    verify(updatesReceiverMock).onFloatGroupPropSetterCalled(1, 0.0f);
    verifyNoMoreInteractions(updatesReceiverMock);
    reset(updatesReceiverMock);
  }
} 