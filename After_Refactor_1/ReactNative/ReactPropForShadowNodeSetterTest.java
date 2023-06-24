/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.uimanager;

import androidx.annotation.Nullable;

import com.facebook.react.bridge.JavaOnlyMap;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.uimanager.annotations.ReactProp;
import com.facebook.react.uimanager.annotations.ReactPropGroup;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;
import org.powermock.core.classloader.annotations.PowerMockIgnore;
import org.powermock.modules.junit4.rule.PowerMockRule;

import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;

/**
 * Test {@link ReactProp} annotation for {@link ReactShadowNode}.
 */
@RunWith(MockitoJUnitRunner.class)
@PowerMockIgnore({"org.mockito.*", "org.robolectric.*", "androidx.*", "android.*"})
@Ignore // TODO T14964130
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

        private ViewManagerUpdatesReceiver mViewManagerUpdatesReceiver;

        private ShadowViewUnderTest(ViewManagerUpdatesReceiver viewManagerUpdatesReceiver) {
            mViewManagerUpdatesReceiver = viewManagerUpdatesReceiver;
        }

        @ReactProp(name = "boolProp")
        public void setBoolProp(boolean value) {
            mViewManagerUpdatesReceiver.onBooleanSetterCalled(value);
        }

        @ReactProp(name = "stringProp")
        public void setStringProp(@Nullable String value) {
            mViewManagerUpdatesReceiver.onStringSetterCalled(value);
        }

        @ReactPropGroup(
                names = {
                        "floatGroupPropFirst",
                        "floatGroupPropSecond",
                })
        public void setFloatGroupProp(int index, float value) {
            mViewManagerUpdatesReceiver.onFloatGroupPropSetterCalled(index, value);
        }
    }

    @Rule
    public PowerMockRule rule = new PowerMockRule();

    @Mock
    private ViewManagerUpdatesReceiver mUpdatesReceiverMock;

    private ShadowViewUnderTest mShadowView;

    @Before
    public void setup() {
        mShadowView = new ShadowViewUnderTest(mUpdatesReceiverMock);
    }

    private void verifySetter(String propName, Object propValue, Runnable verifyAction) {
        mShadowView.updateProperties(buildStyles(propName, propValue));
        verifyAction.run();
        verifyNoMoreInteractions(mUpdatesReceiverMock);
    }

    @Test
    public void testBooleanSetter() {
        verifySetter("boolProp", true, () -> verify(mUpdatesReceiverMock).onBooleanSetterCalled(true));
        verifySetter("boolProp", false, () -> verify(mUpdatesReceiverMock).onBooleanSetterCalled(false));
        verifySetter("boolProp", null, () -> verify(mUpdatesReceiverMock).onBooleanSetterCalled(false));
    }

    @Test
    public void testStringSetter() {
        verifySetter("stringProp", "someRandomString", () -> verify(mUpdatesReceiverMock).onStringSetterCalled("someRandomString"));
        verifySetter("stringProp", null, () -> verify(mUpdatesReceiverMock).onStringSetterCalled(null));
    }

    @Test
    public void testFloatGroupSetter() {
        verifySetter("floatGroupPropFirst", 11.0, () -> verify(mUpdatesReceiverMock).onFloatGroupPropSetterCalled(0, 11.0f));
        verifySetter("floatGroupPropSecond", -111.0, () -> verify(mUpdatesReceiverMock).onFloatGroupPropSetterCalled(1, -111.0f));
        verifySetter("floatGroupPropSecond", null, () -> verify(mUpdatesReceiverMock).onFloatGroupPropSetterCalled(1, 0.0f));
    }
}