// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

// This enum defines the different ways in which Yoga can align items within a container
package com.facebook.yoga;

public enum YogaAlign {
  AUTO(0),
  START(1), // align items to the start of the container
  CENTER(2), // center items within the container
  END(3), // align items to the end of the container
  STRETCH(4), // stretch items to fill the container
  BASELINE(5), // align items to their baseline
  SPACE_BETWEEN(6), // evenly distribute items with space between them
  SPACE_AROUND(7); // evenly distribute items with space around them

  private final int value;

  YogaAlign(int value) {
    this.value = value;
  }

  public int value() {
    return value;
  }

  public static YogaAlign fromValue(int value) {
    switch (value) {
      case 0:
        return AUTO;
      case 1:
        return START;
      case 2:
        return CENTER;
      case 3:
        return END;
      case 4:
        return STRETCH;
      case 5:
        return BASELINE;
      case 6:
        return SPACE_BETWEEN;
      case 7:
        return SPACE_AROUND;
      default:
        throw new IllegalArgumentException("Invalid value: " + value);
    }
  }
}

