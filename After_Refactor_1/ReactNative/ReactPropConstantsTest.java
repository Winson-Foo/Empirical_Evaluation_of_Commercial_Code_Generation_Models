package com.facebook.react.uimanager;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.fail;

import android.view.View;

import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.common.MapBuilder;
import com.facebook.react.uimanager.annotations.ReactProp;
import com.facebook.react.uimanager.annotations.ReactPropGroup;
import com.facebook.testing.robolectric.v3.WithTestDefaultsRunner;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.robolectric.RuntimeEnvironment;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

@RunWith(WithTestDefaultsRunner.class)
public class ReactPropConstantsTest {

  @Mock View view;

  private UIManagerModule uiManagerModule;
  private ReactApplicationContext reactContext;

  @Before
  public void setUp() {
    MockitoAnnotations.initMocks(this);
    ViewManagerUnderTest viewManager = new ViewManagerUnderTest();

    List<ViewManager> viewManagers = Arrays.asList(viewManager);
    reactContext = new ReactApplicationContext(RuntimeEnvironment.application);
    uiManagerModule = new UIManagerModule(reactContext, viewManagers, 0);
  }

  @Test
  public void testBooleanProp() {
    verifyPropType("boolProp", "boolean");
  }

  @Test
  public void testIntegerProp() {
    verifyPropType("intProp", "number");
  }

  @Test
  public void testDoubleProp() {
    verifyPropType("doubleProp", "number");
  }

  @Test
  public void testFloatProp() {
    verifyPropType("floatProp", "number");
  }

  @Test
  public void testStringProp() {
    verifyPropType("stringProp", "String");
  }

  @Test
  public void testBoxedBooleanProp() {
    verifyPropType("boxedBoolProp", "boolean");
  }

  @Test
  public void testBoxedIntegerProp() {
    verifyPropType("boxedIntProp", "number");
  }

  @Test
  public void testArrayProp() {
    verifyPropType("arrayProp", "Array");
  }

  @Test
  public void testMapProp() {
    verifyPropType("mapProp", "Map");
  }

  @Test
  public void testFloatGroupProp() {
    verifyPropType("floatGroupPropFirst", "number");
    verifyPropType("floatGroupPropSecond", "number");
  }

  @Test
  public void testIntegerGroupProp() {
    verifyPropType("intGroupPropFirst", "number");
    verifyPropType("intGroupPropSecond", "number");
  }

  @Test
  public void testBoxedIntegerGroupProp() {
    verifyPropType("boxedIntGroupPropFirst", "number");
    verifyPropType("boxedIntGroupPropSecond", "number");
  }

  @Test
  public void testCustomIntProp() {
    verifyPropType("customIntProp", "date");
  }

  @Test
  public void testCustomBoxedIntegerGroupProp() {
    verifyPropType("customBoxedIntGroupPropFirst", "color");
    verifyPropType("customBoxedIntGroupPropSecond", "color");
  }

  private void verifyPropType(String propName, String expectedPropType) {
    Map<String, String> nativeProps =
        (Map<String, String>) valueAtPath(uiManagerModule.getConstants(), "SomeView", "NativeProps");
    assertThat(nativeProps).containsEntry(propName, expectedPropType);
  }

  private static Object valueAtPath(Map<?, ?> nestedMap, String... keyPath) {
    for (String key : keyPath) {
      assertThat(nestedMap).isInstanceOf(Map.class);
      assertThat(nestedMap).containsKey(key);
      nestedMap = (Map<?, ?>) nestedMap.get(key);
    }
    return nestedMap;
  }

  private class ViewManagerUnderTest extends ViewManager<View, ReactShadowNode> {

    @Override
    public String getName() {
      return "SomeView";
    }

    @Override
    public ReactShadowNode createShadowNodeInstance() {
      fail("This method should not be executed as a part of this test");
      return null;
    }

    @Override
    protected View createViewInstance(ThemedReactContext reactContext) {
      fail("This method should not be executed as a part of this test");
      return null;
    }

    @Override
    public Class<? extends ReactShadowNode> getShadowNodeClass() {
      return ReactShadowNode.class;
    }

    @Override
    public void updateExtraData(View root, Object extraData) {
      fail("This method should not be executed as a part of this test");
    }

    @ReactProp(name = "boolProp")
    public void setBoolProp(View v, boolean value) {}

    @ReactProp(name = "intProp")
    public void setIntProp(View v, int value) {}

    @ReactProp(name = "floatProp")
    public void setFloatProp(View v, float value) {}

    @ReactProp(name = "doubleProp")
    public void setDoubleProp(View v, double value) {}

    @ReactProp(name = "stringProp")
    public void setStringProp(View v, String value) {}

    @ReactProp(name = "boxedBoolProp")
    public void setBoxedBoolProp(View v, Boolean value) {}

    @ReactProp(name = "boxedIntProp")
    public void setBoxedIntProp(View v, Integer value) {}

    @ReactProp(name = "arrayProp")
    public void setArrayProp(View v, ReadableArray value) {}

    @ReactProp(name = "mapProp")
    public void setMapProp(View v, ReadableMap value) {}

    @ReactPropGroup(
        names = {
          "floatGroupPropFirst",
          "floatGroupPropSecond",
        })
    public void setFloatGroupProp(View v, int index, float value) {}

    @ReactPropGroup(names = {"intGroupPropFirst", "intGroupPropSecond"})
    public void setIntGroupProp(View v, int index, int value) {}

    @ReactPropGroup(
        names = {
          "boxedIntGroupPropFirst",
          "boxedIntGroupPropSecond",
        })
    public void setBoxedIntGroupProp(View v, int index, Integer value) {}

    @ReactProp(name = "customIntProp", customType = "date")
    public void customIntProp(View v, int value) {}

    @ReactPropGroup(
        names = {"customBoxedIntGroupPropFirst", "customBoxedIntGroupPropSecond"},
        customType = "color")
    public void customIntGroupProp(View v, int index, Integer value) {}
  }
}