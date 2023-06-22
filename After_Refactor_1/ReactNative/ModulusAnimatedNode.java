/**
 * Represents a ModulusAnimatedNode that calculates the modulus of its input node's value.
 */
class ModulusAnimatedNode extends ValueAnimatedNode {

  private final NativeAnimatedNodesManager mNativeAnimatedNodesManager;
  private final int mInputNodeId;
  private final double mModulus;

  private static final String INPUT_KEY = "input";
  private static final String MODULUS_KEY = "modulus";
  private static final String ERROR_MSG = "Illegal node ID set as an input for Animated.modulus node";

  /**
   * Constructor for ModulusAnimatedNode.
   *
   * @param config                    The configuration for the node.
   * @param nativeAnimatedNodesManager The manager for the animated nodes.
   */
  public ModulusAnimatedNode(
      ReadableMap config, NativeAnimatedNodesManager nativeAnimatedNodesManager) {
    mNativeAnimatedNodesManager = nativeAnimatedNodesManager;
    mInputNodeId = config.getInt(INPUT_KEY);
    mModulus = config.getDouble(MODULUS_KEY);
  }

  /**
   * Updates the value of the node by calculating the modulus of its input node's value.
   */
  @Override
  public void update() {
    AnimatedNode inputNode = mNativeAnimatedNodesManager.getNodeById(mInputNodeId);
    if (inputNode instanceof ValueAnimatedNode) {
      double inputNodeValue = ((ValueAnimatedNode) inputNode).getValue();
      mValue = (inputNodeValue % mModulus + mModulus) % mModulus;
    } else {
        // If the input node is not a ValueAnimatedNode, log an error message instead of throwing an exception.
      Log.e("ModulusAnimatedNode", ERROR_MSG);
    }
  }

  /**
   * Returns a string representation of the node.
   *
   * @return A string containing the node's ID, input node ID, modulus, and value.
   */
  public String toString() {
    return "ModulusAnimatedNode[" + mTag + "] InputNode: " + mInputNodeId + " Modulus: " + mModulus + " Value: " + mValue;
  }
}