import { VRMExpressionPresetName, VRMHumanBoneName } from '@pixiv/three-vrm';

/**
 * VRMCVRMAnimation interface for describing VRM animations.
 */
export interface VRMCVRMAnimation {
  specVersion: string;
  humanoid: HumanoidAnimation;
  expressions?: ExpressionsAnimation;
  lookAt?: LookAtAnimation;
}

/**
 * HumanoidAnimation interface for describing humanoid bone animations.
 */
export interface HumanoidAnimation {
  humanBones: {
    [name in VRMHumanBoneName]?: BoneAnimation;
  };
}

/**
 * ExpressionsAnimation interface for describing expression animations.
 */
export interface ExpressionsAnimation {
  preset?: {
    [name in VRMExpressionPresetName]?: BoneAnimation;
  };
  custom?: {
    [name: string]: BoneAnimation;
  };
}

/**
 * LookAtAnimation interface for describing lookAt animations.
 */
export interface LookAtAnimation {
  node: number;
}

/**
 * BoneAnimation interface for describing bone animations.
 */
export interface BoneAnimation {
  node: number;
}