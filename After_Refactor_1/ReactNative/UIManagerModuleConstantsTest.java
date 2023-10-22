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
public class UIManagerModuleConstantsTest {

  private static final String CUSTOM_BUBBLING_EVENT_TYPES = "customBubblingEventTypes";
  private static final String CUSTOM_DIRECT_EVENT_TYPES = "customDirectEventTypes";
  private static final String RED_PANDA_PHOTO_VIEW = "RedPandaPhotoOfTheDayView";
  private static final String PHOTO_SIZE_TYPE = "PhotoSizeType";

  @Rule public PowerMockRule rule = new PowerMockRule();

  private ReactApplicationContext mReactContext;

  @Before
  public void setUp() {
    mReactContext = new ReactApplicationContext(RuntimeEnvironment.application);
  }

  @Test
  public void shouldReturnConstantsWithoutCustomEventTypes() {
    // Given
    List<ViewManager> viewManagers = Arrays.asList(mock(ViewManager.class));
    UIManagerModule uiManager = new UIManagerModule(mReactContext, viewManagers, 0);
    // When
    Map<String, Object> constants = uiManager.getConstants();
    // Then
    assertThat(constants)
        .containsKey(CUSTOM_BUBBLING_EVENT_TYPES)
        .containsKey(CUSTOM_DIRECT_EVENT_TYPES)
        .containsKey("Dimensions");
  }

  @Test
  public void shouldIncludeCustomBubblingEventTypes() {
    // Given
    ViewManager mockViewManager = mock(ViewManager.class);
    List<ViewManager> viewManagers = Arrays.asList(mockViewManager);
    when(mockViewManager.getExportedCustomBubblingEventTypeConstants())
        .thenReturn(MapBuilder.of("onTwirl", TWIRL_BUBBLING_EVENT_MAP));
    UIManagerModule uiManager = new UIManagerModule(mReactContext, viewManagers, 0);
    // When
    Map<String, Object> constants = uiManager.getConstants();
    // Then
    assertThat((Map) constants.get(CUSTOM_BUBBLING_EVENT_TYPES))
        .contains(MapEntry.entry("onTwirl", TWIRL_BUBBLING_EVENT_MAP))
        .containsKey("topChange");
  }

  @Test
  public void shouldIncludeCustomDirectEventTypes() {
    // Given
    ViewManager mockViewManager = mock(ViewManager.class);
    List<ViewManager> viewManagers = Arrays.asList(mockViewManager);
    when(mockViewManager.getExportedCustomDirectEventTypeConstants())
        .thenReturn(MapBuilder.of("onTwirl", TWIRL_DIRECT_EVENT_MAP));
    UIManagerModule uiManager = new UIManagerModule(mReactContext, viewManagers, 0);
    // When
    Map<String, Object> constants = uiManager.getConstants();
    // Then
    assertThat((Map) constants.get(CUSTOM_DIRECT_EVENT_TYPES))
        .contains(MapEntry.entry("onTwirl", TWIRL_DIRECT_EVENT_MAP))
        .containsKey("topLoadingStart");
  }

  @Test
  public void shouldIncludeCustomViewConstants() {
    // Given
    ViewManager mockViewManager = mock(ViewManager.class);
    List<ViewManager> viewManagers = Arrays.asList(mockViewManager);
    when(mockViewManager.getName()).thenReturn(RED_PANDA_PHOTO_VIEW);
    when(mockViewManager.getExportedViewConstants())
        .thenReturn(
            MapBuilder.of(
                PHOTO_SIZE_TYPE, MapBuilder.of("Small", 1, "Large", 2)));
    UIManagerModule uiManager = new UIManagerModule(mReactContext, viewManagers, 0);
    // When
    Map<String, Object> constants = uiManager.getConstants();
    // Then
    assertThat(constants).containsKey(RED_PANDA_PHOTO_VIEW);
    assertThat((Map) constants.get(RED_PANDA_PHOTO_VIEW)).containsKey("Constants");
    assertThat((Map) valueAtPath(constants, RED_PANDA_PHOTO_VIEW, "Constants"))
        .containsKey(PHOTO_SIZE_TYPE);
  }

  @Test
  public void shouldIncludeNativeProps() {
    // Given
    ViewManager mockViewManager = mock(ViewManager.class);
    List<ViewManager> viewManagers = Arrays.asList(mockViewManager);
    when(mockViewManager.getName()).thenReturn("SomeView");
    when(mockViewManager.getNativeProps()).thenReturn(MapBuilder.of("fooProp", "number"));
    UIManagerModule uiManager = new UIManagerModule(mReactContext, viewManagers, 0);
    // When
    Map<String, Object> constants = uiManager.getConstants();
    // Then
    assertThat((String) valueAtPath(constants, "SomeView", "NativeProps", "fooProp"))
        .isEqualTo("number");
  }

  @Test
  public void shouldMergeConstants() {
    // Given
    ViewManager managerX = mock(ViewManager.class);
    when(managerX.getExportedCustomDirectEventTypeConstants())
        .thenReturn(
            MapBuilder.of(
                "onTwirl",
                MapBuilder.of(
                    "registrationName",
                    "onTwirl",
                    "keyToOverride",
                    "valueX",
                    "mapToMerge",
                    MapBuilder.of("keyToOverride", "innerValueX", "anotherKey", "valueX"))));

    ViewManager managerY = mock(ViewManager.class);
    when(managerY.getExportedCustomDirectEventTypeConstants())
        .thenReturn(
            MapBuilder.of(
                "onTwirl",
                MapBuilder.of(
                    "extraKey",
                    "extraValue",
                    "keyToOverride",
                    "valueY",
                    "mapToMerge",
                    MapBuilder.of("keyToOverride", "innerValueY", "extraKey", "valueY"))));

    List<ViewManager> viewManagers = Arrays.asList(managerX, managerY);
    UIManagerModule uiManager = new UIManagerModule(mReactContext, viewManagers, 0);
    // When
    Map<String, Object> constants = uiManager.getConstants();
    // Then
    assertThat((Map) constants.get(CUSTOM_DIRECT_EVENT_TYPES)).containsKey("onTwirl");

    Map twirlMap = (Map) valueAtPath(constants, CUSTOM_DIRECT_EVENT_TYPES, "onTwirl");
    assertThat(twirlMap)
        .contains(MapEntry.entry("registrationName", "onTwirl"))
        .contains(MapEntry.entry("keyToOverride", "valueY"))
        .contains(MapEntry.entry("extraKey", "extraValue"))
        .containsKey("mapToMerge");

    Map mapToMerge = (Map) valueAtPath(twirlMap, "mapToMerge");
    assertThat(mapToMerge)
        .contains(MapEntry.entry("keyToOverride", "innerValueY"))
        .contains(MapEntry.entry("anotherKey", "valueX"))
        .contains(MapEntry.entry("extraKey", "valueY"));
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