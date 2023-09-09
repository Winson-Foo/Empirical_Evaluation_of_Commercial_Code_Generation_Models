package com.facebook.react.uimanager;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.common.MapBuilder;
import org.assertj.core.data.MapEntry;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

@DisplayName("UIManagerModuleConstantsTest")
class UIManagerModuleConstantsTest {

  private static final String CUSTOM_BUBBLING_EVENT_TYPES = "customBubblingEventTypes";
  private static final String CUSTOM_DIRECT_EVENT_TYPES = "customDirectEventTypes";

  private static final Map<String, Object> TWIRL_BUBBLING_EVENT_MAP =
      MapBuilder.of(
          "phasedRegistrationNames",
          MapBuilder.of("bubbled", "onTwirl", "captured", "onTwirlCaptured"));
  private static final Map<String, Object> TWIRL_DIRECT_EVENT_MAP =
      MapBuilder.of("registrationName", "onTwirl");

  private ReactApplicationContext reactApplicationContext;

  @BeforeEach
  void setUp() {
    reactApplicationContext = new ReactApplicationContext();
  }

  @Test
  @DisplayName("Get constants with no custom constants")
  void testNoCustomConstants() {
    // Arrange
    List<ViewManager> viewManagers = Arrays.asList(mock(ViewManager.class));
    UIManagerModule uiManagerModule = new UIManagerModule(reactApplicationContext, viewManagers, 0);

    // Act
    Map<String, Object> constants = uiManagerModule.getConstants();

    // Assert
    assertThat(constants)
        .containsKey(CUSTOM_BUBBLING_EVENT_TYPES)
        .containsKey(CUSTOM_DIRECT_EVENT_TYPES)
        .containsKey("Dimensions");
  }

  @Test
  @DisplayName("Get constants with custom bubbling events")
  void testCustomBubblingEvents() {
    // Arrange
    ViewManager viewManager = mock(ViewManager.class);
    when(viewManager.getExportedCustomBubblingEventTypeConstants())
        .thenReturn(MapBuilder.of("onTwirl", TWIRL_BUBBLING_EVENT_MAP));
    List<ViewManager> viewManagers = Arrays.asList(viewManager);
    UIManagerModule uiManagerModule = new UIManagerModule(reactApplicationContext, viewManagers, 0);

    // Act
    Map<String, Object> constants = uiManagerModule.getConstants();

    // Assert
    assertThat((Map<?, ?>) constants.get(CUSTOM_BUBBLING_EVENT_TYPES))
        .contains(MapEntry.entry("onTwirl", TWIRL_BUBBLING_EVENT_MAP))
        .containsKey("topChange");
  }

  @Test
  @DisplayName("Get constants with custom direct events")
  void testCustomDirectEvents() {
    // Arrange
    ViewManager viewManager = mock(ViewManager.class);
    when(viewManager.getExportedCustomDirectEventTypeConstants())
        .thenReturn(MapBuilder.of("onTwirl", TWIRL_DIRECT_EVENT_MAP));
    List<ViewManager> viewManagers = Arrays.asList(viewManager);
    UIManagerModule uiManagerModule = new UIManagerModule(reactApplicationContext, viewManagers, 0);

    // Act
    Map<String, Object> constants = uiManagerModule.getConstants();

    // Assert
    assertThat((Map<?, ?>) constants.get(CUSTOM_DIRECT_EVENT_TYPES))
        .contains(MapEntry.entry("onTwirl", TWIRL_DIRECT_EVENT_MAP))
        .containsKey("topLoadingStart");
  }

  @Test
  @DisplayName("Get constants with custom view constants")
  void testCustomViewConstants() {
    // Arrange
    ViewManager viewManager = mock(ViewManager.class);
    when(viewManager.getName()).thenReturn("RedPandaPhotoOfTheDayView");
    when(viewManager.getExportedViewConstants())
        .thenReturn(MapBuilder.of("PhotoSizeType", MapBuilder.of("Small", 1, "Large", 2)));
    List<ViewManager> viewManagers = Arrays.asList(viewManager);
    UIManagerModule uiManagerModule = new UIManagerModule(reactApplicationContext, viewManagers, 0);

    // Act
    Map<String, Object> constants = uiManagerModule.getConstants();

    // Assert
    assertThat(constants).containsKey("RedPandaPhotoOfTheDayView");
    assertThat((Map<?, ?>) constants.get("RedPandaPhotoOfTheDayView")).containsKey("Constants");
    assertThat((Map<?, ?>) valueAtPath(constants, "RedPandaPhotoOfTheDayView", "Constants"))
        .containsKey("PhotoSizeType");
  }

  @Test
  @DisplayName("Get constants with native props")
  void testNativeProps() {
    // Arrange
    ViewManager viewManager = mock(ViewManager.class);
    when(viewManager.getName()).thenReturn("SomeView");
    when(viewManager.getNativeProps()).thenReturn(MapBuilder.of("fooProp", "number"));
    List<ViewManager> viewManagers = Arrays.asList(viewManager);
    UIManagerModule uiManagerModule = new UIManagerModule(reactApplicationContext, viewManagers, 0);

    // Act
    Map<String, Object> constants = uiManagerModule.getConstants();

    // Assert
    assertThat((String) valueAtPath(constants, "SomeView", "NativeProps", "fooProp"))
        .isEqualTo("number");
  }

  @ParameterizedTest
  @DisplayName("Merge constants")
  @MethodSource("provideMergeConstantParameters")
  void testMergeConstants(
      Map<String, Object> managerXEventConstants,
      Map<String, Object> managerYEventConstants,
      Map<String, Object> expectedEventConstants) {
    // Arrange
    ViewManager viewManagerX = mock(ViewManager.class);
    ViewManager viewManagerY = mock(ViewManager.class);
    when(viewManagerX.getExportedCustomDirectEventTypeConstants()).thenReturn(managerXEventConstants);
    when(viewManagerY.getExportedCustomDirectEventTypeConstants()).thenReturn(managerYEventConstants);
    List<ViewManager> viewManagers = Arrays.asList(viewManagerX, viewManagerY);
    UIManagerModule uiManagerModule = new UIManagerModule(reactApplicationContext, viewManagers, 0);

    // Act
    Map<String, Object> constants = uiManagerModule.getConstants();

    // Assert
    assertThat((Map<?, ?>) constants.get(CUSTOM_DIRECT_EVENT_TYPES)).isEqualTo(expectedEventConstants);
  }

  static Stream<Arguments> provideMergeConstantParameters() {
    Map<String, Object> xEventConstants =
        MapBuilder.of(
            "onTwirl",
            MapBuilder.of(
                "registrationName",
                "onTwirl",
                "keyToOverride",
                "valueX",
                "mapToMerge",
                MapBuilder.of("keyToOverride", "innerValueX", "anotherKey", "valueX")));
    Map<String, Object> yEventConstants =
        MapBuilder.of(
            "onTwirl",
            MapBuilder.of(
                "extraKey",
                "extraValue",
                "keyToOverride",
                "valueY",
                "mapToMerge",
                MapBuilder.of("keyToOverride", "innerValueY", "extraKey", "valueY")));
    Map<String, Object> expectedEventConstants =
        MapBuilder.of(
            "onTwirl",
            MapBuilder.of(
                "registrationName",
                "onTwirl",
                "extraKey",
                "extraValue",
                "keyToOverride",
                "valueY",
                "mapToMerge",
                MapBuilder.of(
                    "keyToOverride", "innerValueY", "anotherKey", "valueX", "extraKey", "valueY")));

    return Stream.of(
        Arguments.of(xEventConstants, yEventConstants, expectedEventConstants),
        Arguments.of(MapBuilder.emptyMap(), yEventConstants, yEventConstants),
        Arguments.of(xEventConstants, MapBuilder.emptyMap(), xEventConstants),
        Arguments.of(MapBuilder.emptyMap(), MapBuilder.emptyMap(), MapBuilder.emptyMap()));
  }

  private static Object valueAtPath(Map<?, ?> nestedMap, String... keyPath) {
    assertThat(keyPath).isNotEmpty();
    Object value = nestedMap;
    for (String key : keyPath) {
      assertThat(value).isInstanceOf(Map.class);
      nestedMap = (Map<?, ?>) value;
      assertThat(nestedMap).containsKey(key);
      value = nestedMap.get(key);
    }
    return value;
  }
} 