/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.animated;

import com.facebook.react.bridge.JSApplicationCausedNativeException;
import com.facebook.react.bridge.ReadableMap;

/*package*/ class ModulusAnimatedNode extends ValueAnimatedNode {
  
  private final NativeAnimatedNodesManager nativeAnimatedNodesManager;
  private final int inputNodeTag;
  private final double modulusValue;
  private static final String INPUT_FIELD = "input";
  private static final String MODULUS_FIELD = "modulus";
  
  public ModulusAnimatedNode(ReadableMap config, NativeAnimatedNodesManager nodesManager) {
    nativeAnimatedNodesManager = nodesManager;
    inputNodeTag = config.getInt(INPUT_FIELD);
    modulusValue = config.getDouble(MODULUS_FIELD);
  }

  @Override
  public void update() {
    AnimatedNode inputNode = nativeAnimatedNodesManager.getNodeById(inputNodeTag);
    if (inputNode != null && inputNode instanceof ValueAnimatedNode) {
      double inputValue = ((ValueAnimatedNode) inputNode).getValue();
      mValue = (inputValue % modulusValue + modulusValue) % modulusValue;
    } else {
      throw new JSApplicationCausedNativeException(
          "Illegal node ID set as an input for Animated.modulus node");
    }
  }

  // For debugging purposes
  public String prettyPrint() {
    return "ModulusAnimatedNode["
        + mTag
        + "] inputNodeTag: "
        + inputNodeTag
        + " modulusValue: "
        + modulusValue
        + " super: "
        + super.prettyPrint();
  }
}

