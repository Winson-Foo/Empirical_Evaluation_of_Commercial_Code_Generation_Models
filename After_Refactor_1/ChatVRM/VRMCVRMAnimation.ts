import { VRMExpressionPresetName, VRMHumanBoneName } from '@pixiv/three-vrm';

type HumanBone = {
  node: number;
}

type ExpressionPreset = {
  node: number;
}

type VRMCHumanoid = {
  humanBones: Record<VRMHumanBoneName, HumanBone | undefined>;
}

type VRMCExpressions = {
  preset?: Record<VRMExpressionPresetName, ExpressionPreset | undefined>;
  custom?: Record<string, ExpressionPreset>;
}

type VRMCLookAt = {
  node: number;
}

export interface VRMCVRMAnimation {
  specVersion: string;
  humanoid: VRMCHumanoid;
  expressions?: VRMCExpressions;
  lookAt?: VRMCLookAt;
}