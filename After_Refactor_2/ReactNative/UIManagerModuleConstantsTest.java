/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.uimanager;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.common.MapBuilder;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.assertj.core.data.MapEntry;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.powermock.core.classloader.annotations.PowerMockIgnore;
import org.powermock.modules.junit4.rule.PowerMockRule;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.RuntimeEnvironment;

@RunWith(RobolectricTestRunner.class)
@PowerMockIgnore({"org.mockito.*", "org.robolectric.*", "androidx.*", "android.*"})
@Ignore // TODO T14964130
public class ViewManagerTest {

  @Rule public PowerMockRule rule = new PowerMockRule();

  private static final String CUSTOM_BUBBLING_EVENT_TYPES = "customBubblingEventTypes";
  private static final String CUSTOM_DIRECT_EVENT_TYPES = "customDirectEventTypes";
  private static final String VIEW_CONSTANTS = "Constants";
  private static final String DIMENSIONS = "Dimensions";
  private static final String VIEW_NAME = "RedPandaPhotoOfTheDayView";
  private static final String NATIVE_PROPS = "NativeProps";
  private static final String ON_TWIRL = "onTwirl";
  private static final String MAP_TO_MERGE = "mapToMerge";
  private static final String REGISTRATION_NAME = "registrationName";
  private static final String TOP_CHANGE = "topChange";
  private static final String KEY_TO_OVERRIDE = "keyToOverride";
  private static final String EXTRA_KEY = "extraKey";
  private static final String PHASED_REGISTRATION_NAMES = "phasedRegistrationNames";
  private static final String TOP_LOADING_START = "topLoadingStart";

  private static final Map TWIRL_BUBBLING_EVENT_MAP =
      MapBuilder.of(
          PHASED_REGISTRATION_NAMES,
          MapBuilder.of("bubbled", ON_TWIRL, "captured", "onTwirlCaptured"));
  private static final Map TWIRL_DIRECT_EVENT_MAP = MapBuilder.of(REGISTRATION_NAME, ON_TWIRL);

  private ReactApplicationContext reactContext;

  @Before
  public void setUp() {
    reactContext = new ReactApplicationContext(RuntimeEnvironment.application);
  }

  @Test
  public void testNoCustomConstants() {
    List<ViewManager> viewManagers = Arrays.asList(mock(ViewManager.class));
    UIManagerModule uiManagerModule = new UIManagerModule(reactContext, viewManagers, 0);
    Map<String, Object> constants = uiManagerModule.getConstants();
    assertThat(constants)
        .containsKey(CUSTOM_BUBBLING_EVENT_TYPES)
        .containsKey(CUSTOM_DIRECT_EVENT_TYPES)
        .containsKey(DIMENSIONS);
  }

  @Test
  public void testCustomBubblingEvents() {
    ViewManager mockViewManager = mock(ViewManager.class);
    List<ViewManager> viewManagers = Arrays.asList(mockViewManager);
    when(mockViewManager.getExportedCustomBubblingEventTypeConstants())
        .thenReturn(MapBuilder.of(ON_TWIRL, TWIRL_BUBBLING_EVENT_MAP));
    UIManagerModule uiManagerModule = new UIManagerModule(reactContext, viewManagers, 0);
    Map<String, Object> constants = uiManagerModule.getConstants();
    assertThat((Map) constants.get(CUSTOM_BUBBLING_EVENT_TYPES))
        .contains(MapEntry.entry(ON_TWIRL, TWIRL_BUBBLING_EVENT_MAP))
        .containsKey(TOP_CHANGE);
  }

  @Test
  public void testCustomDirectEvents() {
    ViewManager mockViewManager = mock(ViewManager.class);
    List<ViewManager> viewManagers = Arrays.asList(mockViewManager);
    when(mockViewManager.getExportedCustomDirectEventTypeConstants())
        .thenReturn(MapBuilder.of(ON_TWIRL, TWIRL_DIRECT_EVENT_MAP));
    UIManagerModule uiManagerModule = new UIManagerModule(reactContext, viewManagers, 0);
    Map<String, Object> constants = uiManagerModule.getConstants();
    assertThat((Map) constants.get(CUSTOM_DIRECT_EVENT_TYPES))
        .contains(MapEntry.entry(ON_TWIRL, TWIRL_DIRECT_EVENT_MAP))
        .containsKey(TOP_LOADING_START);
  }

  @Test
  public void testCustomViewConstants() {
    ViewManager mockViewManager = mock(ViewManager.class);
    List<ViewManager> viewManagers = Arrays.asList(mockViewManager);
    when(mockViewManager.getName()).thenReturn(VIEW_NAME);
    when(mockViewManager.getExportedViewConstants())
        .thenReturn(MapBuilder.of("PhotoSizeType", MapBuilder.of("Small", 1, "Large", 2)));
    UIManagerModule uiManagerModule = new UIManagerModule(reactContext, viewManagers, 0);
    Map<String, Object> constants = uiManagerModule.getConstants();
    assertThat(constants).containsKey(VIEW_NAME);
    assertThat((Map) constants.get(VIEW_NAME)).containsKey(VIEW_CONSTANTS);
    assertThat((Map) valueAtPath(constants, VIEW_NAME, VIEW_CONSTANTS))
        .containsKey("PhotoSizeType");
  }

  @Test
  public void testNativeProps() {
    ViewManager mockViewManager = mock(ViewManager.class);
    List<ViewManager> viewManagers = Arrays.asList(mockViewManager);
    when(mockViewManager.getName()).thenReturn("SomeView");
    when(mockViewManager.getNativeProps()).thenReturn(MapBuilder.of("fooProp", "number"));
    UIManagerModule uiManagerModule = new UIManagerModule(reactContext, viewManagers, 0);
    Map<String, Object> constants = uiManagerModule.getConstants();
    assertThat((String) valueAtPath(constants, "SomeView", NATIVE_PROPS, "fooProp"))
        .isEqualTo("number");
  }

  @Test
  public void testMergeConstants() {
    ViewManager managerX = mock(ViewManager.class);
    when(managerX.getExportedCustomDirectEventTypeConstants())
        .thenReturn(
            MapBuilder.of(
                ON_TWIRL,
                MapBuilder.of(
                    REGISTRATION_NAME,
                    ON_TWIRL,
                    KEY_TO_OVERRIDE,
                    "valueX",
                    MAP_TO_MERGE,
                    MapBuilder.of(KEY_TO_OVERRIDE, "innerValueX", EXTRA_KEY, "valueX"))));

    ViewManager managerY = mock(ViewManager.class);
    when(managerY.getExportedCustomDirectEventTypeConstants())
        .thenReturn(
            MapBuilder.of(
                ON_TWIRL,
                MapBuilder.of(
                    EXTRA_KEY,
                    "extraValue",
                    KEY_TO_OVERRIDE,
                    "valueY",
                    MAP_TO_MERGE,
                    MapBuilder.of(KEY_TO_OVERRIDE, "innerValueY", EXTRA_KEY, "valueY"))));

    List<ViewManager> viewManagers = Arrays.asList(managerX, managerY);
    UIManagerModule uiManagerModule = new UIManagerModule(reactContext, viewManagers, 0);
    Map<String, Object> constants = uiManagerModule.getConstants();
    assertThat((Map) constants.get(CUSTOM_DIRECT_EVENT_TYPES)).containsKey(ON_TWIRL);

    Map twirlMap = (Map) valueAtPath(constants, CUSTOM_DIRECT_EVENT_TYPES, ON_TWIRL);
    assertThat(twirlMap)
        .contains(MapEntry.entry(REGISTRATION_NAME, ON_TWIRL))
        .contains(MapEntry.entry(KEY_TO_OVERRIDE, "valueY"))
        .contains(MapEntry.entry(EXTRA_KEY, "extraValue"))
        .containsKey(MAP_TO_MERGE);

    Map mapToMerge = (Map) valueAtPath(twirlMap, MAP_TO_MERGE);
    assertThat(mapToMerge)
        .contains(MapEntry.entry(KEY_TO_OVERRIDE, "innerValueY"))
        .contains(MapEntry.entry(EXTRA_KEY, "valueY"))
        .contains(MapEntry.entry("anotherKey", "valueX"));
  }

  private static Object valueAtPath(Map nestedMap, String... keyPath) {
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