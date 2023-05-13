import { VRMExpressionPresetName, VRMHumanBoneName } from '@pixiv/three-vrm';

interface HumanBone {
  readonly node: number;
}

interface HumanBones {
  readonly [name in VRMHumanBoneName]?: HumanBone;
}

interface Expression {
  readonly node: number;
}

interface Expressions {
  readonly preset?: {
    readonly [name in VRMExpressionPresetName]?: Expression;
  };
  readonly custom?: {
    readonly [name: string]: Expression;
  };
}

interface LookAt {
  readonly node: number;
}

export interface VRMCVRMAnimation {
  readonly specVersion: string;
  readonly humanoid: {
    readonly humanBones: HumanBones;
  };
  readonly expressions?: Expressions;
  readonly lookAt?: LookAt;
}