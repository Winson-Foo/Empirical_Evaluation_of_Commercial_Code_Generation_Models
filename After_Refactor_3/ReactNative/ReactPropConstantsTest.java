package com.facebook.react.uimanager;

import android.view.View;

import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.uimanager.annotations.ReactProp;
import com.facebook.react.uimanager.annotations.ReactPropGroup;

public class ViewManagerUnderTest extends ViewManager<View, ReactShadowNode> {

    @Override
    public String getName() {
        return "SomeView";
    }

    @Override
    public ReactShadowNode createShadowNodeInstance() {
        throw new IllegalStateException("This method should not be executed as a part of this test");
    }

    @Override
    protected View createViewInstance(ThemedReactContext reactContext) {
        throw new IllegalStateException("This method should not be executed as a part of this test");
    }

    @Override
    public Class<? extends ReactShadowNode> getShadowNodeClass() {
        return ReactShadowNode.class;
    }

    @Override
    public void updateExtraData(View root, Object extraData) {
        throw new IllegalStateException("This method should not be executed as a part of this test");
    }

    @ReactProp(name = "boolProp")
    public void setBoolProp(View view, boolean value) {}

    @ReactProp(name = "intProp")
    public void setIntProp(View view, int value) {}

    @ReactProp(name = "floatProp")
    public void setFloatProp(View view, float value) {}

    @ReactProp(name = "doubleProp")
    public void setDoubleProp(View view, double value) {}

    @ReactProp(name = "stringProp")
    public void setStringProp(View view, String value) {}

    @ReactProp(name = "boxedBoolProp")
    public void setBoxedBoolProp(View view, Boolean value) {}

    @ReactProp(name = "boxedIntProp")
    public void setBoxedIntProp(View view, Integer value) {}

    @ReactProp(name = "arrayProp")
    public void setArrayProp(View view, ReadableArray value) {}

    @ReactProp(name = "mapProp")
    public void setMapProp(View view, ReadableMap value) {}

    @ReactPropGroup(
            names = {"floatGroupPropFirst", "floatGroupPropSecond"})
    public void setFloatGroupProp(View view, int index, float value) {}

    @ReactPropGroup(names = {"intGroupPropFirst", "intGroupPropSecond"})
    public void setIntGroupProp(View view, int index, int value) {}

    @ReactPropGroup(
            names = {"boxedIntGroupPropFirst", "boxedIntGroupPropSecond"})
    public void setBoxedIntGroupProp(View view, int index, Integer value) {}

    @ReactProp(name = "customIntProp", customType = "date")
    public void setCustomIntProp(View view, int value) {}

    @ReactPropGroup(
            names = {"customBoxedIntGroupPropFirst", "customBoxedIntGroupPropSecond"},
            customType = "color")
    public void setCustomIntGroupProp(View view, int index, Integer value) {}
}
```

**ReactPropConstantsTest.java**
```
package com.facebook.react.uimanager;

import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.common.MapBuilder;
import com.facebook.react.uimanager.ViewManagerUnderTest;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.powermock.core.classloader.annotations.PowerMockIgnore;
import org.powermock.modules.junit4.rule.PowerMockRule;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.RuntimeEnvironment;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

@RunWith(RobolectricTestRunner.class)
@PowerMockIgnore({"org.mockito.*", "org.robolectric.*", "androidx.*", "android.*"})
@Ignore // TODO T14964130
public class ReactPropConstantsTest {

    @Rule
    public PowerMockRule rule = new PowerMockRule();

    @Test
    public void testNativePropsIncludeCorrectTypes() {
        // Arrange
        List<ViewManager> managers = Arrays.asList(new ViewManagerUnderTest());
        ReactApplicationContext context = new ReactApplicationContext(RuntimeEnvironment.application);
        UIManagerModule uiManager = new UIManagerModule(context, managers, 0);

        // Act
        Object value = valueAtPath(uiManager.getConstants(), "SomeView", "NativeProps");
        Map<String, String> constants = (Map) value;

        // Assert
        assertThat(constants).containsExactly(
            MapBuilder.<String, String>builder()
                .put("boolProp", "boolean")
                .put("intProp", "number")
                .put("floatProp", "number")
                .put("doubleProp", "number")
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
                .build()
        );
    }

    private Object valueAtPath(Map nestedMap, String... keyPath) {
        assertThat(keyPath).isNotEmpty();
        Object value = nestedMap;
        for (String key : keyPath) {
            assertThat(value).isInstanceOf(Map.class);
            nestedMap = (Map) value;
            assertThat(nestedMap).containsKey(key);
            value = nestedMap.get(key);
        }
        return value;
    }
} 