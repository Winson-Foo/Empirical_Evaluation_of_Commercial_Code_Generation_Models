package com.facebook.react.animated;

import com.facebook.react.bridge.JSApplicationCausedNativeException;
import com.facebook.react.bridge.ReadableMap;

/**
 * A node that outputs the modulus of a given input node value and a modulus constant
 */
class ModulusAnimatedNode extends ValueAnimatedNode {

  private final NativeAnimatedNodesManager nodesManager;
  private final int inputNodeID;
  private final double modulus;

  public ModulusAnimatedNode(
      ReadableMap config, NativeAnimatedNodesManager nodesManager) {
    this.nodesManager = nodesManager;
    this.inputNodeID = config.getInt("input");
    this.modulus = config.getDouble("modulus");
  }

  /**
   * Updates the output value of the node by computing its modulus with the given input node value
   * and the modulus constant.
   *
   * @throws JSApplicationCausedNativeException if the input node is invalid
   */
  @Override
  public void update() {
    AnimatedNode inputNode = nodesManager.getNodeById(inputNodeID);
    if (inputNode != null && inputNode instanceof ValueAnimatedNode) {
      double inputValue = ((ValueAnimatedNode) inputNode).getValue();
      double outputValue = (inputValue % modulus + modulus) % modulus;
      setValue(outputValue);
    } else {
      throw new JSApplicationCausedNativeException("Invalid input node ID for modulus node");
    }
  }

  /**
   * Returns a string representation of the node for debugging purposes.
   */
  public String toString() {
    return "ModulusAnimatedNode [inputNodeID=" + inputNodeID + ", modulus=" + modulus + "]";
  }
}