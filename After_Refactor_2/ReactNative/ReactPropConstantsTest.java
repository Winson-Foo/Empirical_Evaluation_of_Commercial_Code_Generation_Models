package com.facebook.react.uimanager;

import android.view.View;

import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.common.MapBuilder;
import com.facebook.react.uimanager.annotations.ReactProp;
import com.facebook.react.uimanager.annotations.ReactPropGroup;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.powermock.modules.junit4.PowerMockRunner;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.RuntimeEnvironment;

import static org.assertj.core.api.Assertions.assertThat;

@RunWith(RobolectricTestRunner.class)
public class ReactPropConstantsTest {

  private final String SOME_VIEW_MANAGER_NAME = "SomeView";

  @Test
  public void testNativePropsIncludeCorrectTypes() {
    List<ViewManager> viewManagers = Arrays.<ViewManager>asList(new ViewManagerUnderTest());
    ReactApplicationContext reactContext =
        new ReactApplicationContext(RuntimeEnvironment.application);
    UIManagerModule uiManagerModule = new UIManagerModule(reactContext, viewManagers, 0);
    Map<String, Object> constants = getNativePropsForViewManager(uiManagerModule, SOME_VIEW_MANAGER_NAME);
    assertThat(constants)
        .isEqualTo(
            MapBuilder.<String, Object>builder()
                .put("boolProp", "boolean")
                .put("intProp", "number")
                .put("doubleProp", "number")
                .put("floatProp", "number")
                .put("stringProp", "String")
                .put("boxedBoolProp", "boolean")
                .put("boxedIntProp", "number")
                .put("arrayProp", "Array")
                .put("mapProp", "Map")
                .put("floatGroupPropFirst", "number")
                .put("floatGroupPropSecond", "number")
                .put("intGroupPropFirst", "number")
                .put("intGroupPropSecond", "number")
                .put("boxedIntGroupPropFirst", "number")
                .put("boxedIntGroupPropSecond", "number")
                .put("customIntProp", "date")
                .put("customBoxedIntGroupPropFirst", "color")
                .put("customBoxedIntGroupPropSecond", "color")
                .build());
  }

  private Map<String, Object> getNativePropsForViewManager(UIManagerModule module, String viewManagerName) {
    Object constants = module.getConstants().get(viewManagerName);
    assertThat(constants).isInstanceOf(Map.class);
    Map<String, Object> constantsMap = (Map<String, Object>) constants;
    assertThat(constantsMap).containsKey("NativeProps");
    return (Map<String, Object>) constantsMap.get("NativeProps");
  }

  private class ViewManagerUnderTest extends ViewManager<View, ReactShadowNode> {

    @Override
    public String getName() {
      return SOME_VIEW_MANAGER_NAME;
    }

    @Override
    public ReactShadowNode createShadowNodeInstance() {
      throw new UnsupportedOperationException();
    }

    @Override
    protected View createViewInstance(ThemedReactContext reactContext) {
      throw new UnsupportedOperationException();
    }

    @Override
    public Class<? extends ReactShadowNode> getShadowNodeClass() {
      return ReactShadowNode.class;
    }

    @Override
    public void updateExtraData(View root, Object extraData) {
      throw new UnsupportedOperationException();
    }

    @ReactProp(name = "boolProp")
    public void setBooleanProperty(View view, boolean value) {}

    @ReactProp(name = "intProp")
    public void setIntegerProperty(View view, int value) {}

    @ReactProp(name = "floatProp")
    public void setFloatProperty(View view, float value) {}

    @ReactProp(name = "doubleProp")
    public void setDoubleProperty(View view, double value) {}

    @ReactProp(name = "stringProp")
    public void setStringProperty(View view, String value) {}

    @ReactProp(name = "boxedBoolProp")
    public void setBoxedBooleanProperty(View view, Boolean value) {}

    @ReactProp(name = "boxedIntProp")
    public void setBoxedIntegerProperty(View view, Integer value) {}

    @ReactProp(name = "arrayProp")
    public void setArrayProperty(View view, ReadableArray value) {}

    @ReactProp(name = "mapProp")
    public void setMapProperty(View view, ReadableMap value) {}

    @ReactPropGroup(
        names = {
          "floatGroupPropFirst",
          "floatGroupPropSecond",
        })
    public void setFloatGroupProperty(View view, int index, float value) {}

    @ReactPropGroup(names = {"intGroupPropFirst", "intGroupPropSecond"})
    public void setIntegerGroupProperty(View view, int index, int value) {}

    @ReactPropGroup(
        names = {
          "boxedIntGroupPropFirst",
          "boxedIntGroupPropSecond",
        })
    public void setBoxedIntegerGroupProperty(View view, int index, Integer value) {}

    @ReactProp(name = "customIntProp", customType = "date")
    public void setCustomIntegerProperty(View view, int value) {}

    @ReactPropGroup(
        names = {"customBoxedIntGroupPropFirst", "customBoxedIntGroupPropSecond"},
        customType = "color")
    public void setCustomIntegerGroupProperty(View view, int index, Integer value) {}
  }
} 