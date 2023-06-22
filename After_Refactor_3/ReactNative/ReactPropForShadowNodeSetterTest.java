package com.facebook.react.uimanager;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.reset;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;

import androidx.annotation.Nullable;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.uimanager.annotations.ReactProp;
import com.facebook.react.uimanager.annotations.ReactPropGroup;
import java.util.Map;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

public class ReactShadowNodeImplTest {

  public interface PropSetterReceiver {
    void onBooleanPropSet(boolean value);

    void onIntPropSet(int value);

    void onDoublePropSet(double value);

    void onFloatPropSet(float value);

    void onStringPropSet(String value);

    void onBoxedBooleanPropSet(Boolean value);

    void onBoxedIntPropPropSet(Integer value);

    void onArrayPropSet(ReadableArray value);

    void onMapPropSet(ReadableMap value);

    void onFloatGroupPropFirstSet(float value);

    void onFloatGroupPropSecondSet(float value);

    void onBoxedIntGroupPropSet(int index, Integer value);
  }

  public static ReactStylesDiffMap buildStyles(Map<String, Object> styles) {
    return new ReactStylesDiffMap(styles);
  }

  private static PropSetterReceiver mPropSetterReceiver;
  private ReactShadowNodeImpl mShadowNode;

  @BeforeClass
  public static void setupClass() {
    mPropSetterReceiver = mock(PropSetterReceiver.class);
  }

  @Before
  public void setup() {
    mShadowNode = new ReactShadowNodeImpl();
  }

  @Test
  public void testBooleanPropSet() {
    mShadowNode.updateProperties(buildStyles(Map.of("boolProp", true)));
    verify(mPropSetterReceiver).onBooleanPropSet(true);
    verifyNoMoreInteractions(mPropSetterReceiver);
    reset(mPropSetterReceiver);

    mShadowNode.updateProperties(buildStyles(Map.of("boolProp", false)));
    verify(mPropSetterReceiver).onBooleanPropSet(false);
    verifyNoMoreInteractions(mPropSetterReceiver);
    reset(mPropSetterReceiver);

    mShadowNode.updateProperties(buildStyles(Map.of("boolProp", null)));
    verify(mPropSetterReceiver).onBooleanPropSet(false);
    verifyNoMoreInteractions(mPropSetterReceiver);
    reset(mPropSetterReceiver);
  }

  @Test
  public void testStringPropSet() {
    mShadowNode.updateProperties(buildStyles(Map.of("stringProp", "someRandomString")));
    verify(mPropSetterReceiver).onStringPropSet("someRandomString");
    verifyNoMoreInteractions(mPropSetterReceiver);
    reset(mPropSetterReceiver);

    mShadowNode.updateProperties(buildStyles(Map.of("stringProp", null)));
    verify(mPropSetterReceiver).onStringPropSet(null);
    verifyNoMoreInteractions(mPropSetterReceiver);
    reset(mPropSetterReceiver);
  }

  @Test
  public void testFloatGroupPropSet() {
    mShadowNode.updateProperties(buildStyles(Map.of("floatGroupPropFirst", 11.0)));
    verify(mPropSetterReceiver).onFloatGroupPropFirstSet(11.0f);
    verifyNoMoreInteractions(mPropSetterReceiver);
    reset(mPropSetterReceiver);

    mShadowNode.updateProperties(buildStyles(Map.of("floatGroupPropSecond", -111.0)));
    verify(mPropSetterReceiver).onFloatGroupPropSecondSet(-111.0f);
    verifyNoMoreInteractions(mPropSetterReceiver);
    reset(mPropSetterReceiver);

    mShadowNode.updateProperties(buildStyles(Map.of("floatGroupPropSecond", null)));
    verify(mPropSetterReceiver).onFloatGroupPropSecondSet(0.0f);
    verifyNoMoreInteractions(mPropSetterReceiver);
    reset(mPropSetterReceiver);
  }

  private class ShadowNodeUnderTest extends ReactShadowNodeImpl {

    private PropSetterReceiver mPropSetterReceiver;

    private ShadowNodeUnderTest(PropSetterReceiver propSetterReceiver) {
      mPropSetterReceiver = propSetterReceiver;
    }

    @ReactProp(name = "boolProp")
    public void setBoolProp(boolean value) {
      mPropSetterReceiver.onBooleanPropSet(value);
    }

    @ReactProp(name = "stringProp")
    public void setStringProp(@Nullable String value) {
      mPropSetterReceiver.onStringPropSet(value);
    }

    @ReactProp(name = "boxedIntProp")
    public void setBoxedIntPropProp(@Nullable Integer value) {
      mPropSetterReceiver.onBoxedIntPropPropSet(value);
    }

    @ReactPropGroup(
      names = {
        "floatGroupPropFirst",
        "floatGroupPropSecond",
      }
    )
    public void setFloatGroupPropFirst(float value) {
      mPropSetterReceiver.onFloatGroupPropFirstSet(value);
    }

    @ReactPropGroup(
      names = {
        "floatGroupPropFirst",
        "floatGroupPropSecond",
      }
    )
    public void setFloatGroupPropSecond(float value) {
      mPropSetterReceiver.onFloatGroupPropSecondSet(value);
    }

    @ReactPropGroup(
      names = {
        "boxedIntGroupPropFirst",
        "boxedIntGroupPropSecond",
      }
    )
    public void setBoxedIntGroupProp(int index, Integer value) {
      mPropSetterReceiver.onBoxedIntGroupPropSet(index, value);
    }
  }

} 