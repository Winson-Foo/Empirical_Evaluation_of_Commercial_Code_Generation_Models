/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.yoga;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;

public enum YogaAlign {
  AUTO(0),
  FLEX_START(1),
  CENTER(2),
  FLEX_END(3),
  STRETCH(4),
  BASELINE(5),
  SPACE_BETWEEN(6),
  SPACE_AROUND(7);

  private final int mIntValue;

  private static final Map<Integer, YogaAlign> sIntValueToEnum = new HashMap<>();
  static {
    for (YogaAlign alignEnum : EnumSet.allOf(YogaAlign.class)) {
      sIntValueToEnum.put(alignEnum.intValue(), alignEnum);
    }
  }

  YogaAlign(int intValue) {
    mIntValue = intValue;
  }

  public int intValue() {
    return mIntValue;
  }

  public static YogaAlign fromInt(int value) {
    YogaAlign alignEnum = sIntValueToEnum.get(value);
    if (alignEnum == null) {
      throw new IllegalArgumentException("Unknown enum value: " + value);
    }
    return alignEnum;
  }
}