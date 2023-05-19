import * as THREE from "three";
import {
  GLTF,
  GLTFLoaderPlugin,
  GLTFParser,
} from "three/examples/jsm/loaders/GLTFLoader";
import { GLTF as GLTFSchema } from "@gltf-transform/core";
import { VRMAnimationLoaderPluginOptions } from "./VRMAnimationLoaderPluginOptions";
import { VRMCVRMAnimation } from "./VRMCVRMAnimation";
import { VRMHumanBoneName, VRMHumanBoneParentMap } from "@pixiv/three-vrm";
import { VRMAnimation } from "./VRMAnimation";
import { arrayChunk } from "./utils/arrayChunk";

const MAT4_IDENTITY = new THREE.Matrix4();

const _v3A = new THREE.Vector3();
const _quatA = new THREE.Quaternion();
const _quatB = new THREE.Quaternion();
const _quatC = new THREE.Quaternion();

interface IVRMAnimationLoaderPluginNodeMap {
  humanoidIndexToName: Map<number, VRMHumanBoneName>;
  expressionsIndexToName: Map<number, string>;
  lookAtIndex: number | null;
}

interface IVRMAnimationLoaderPluginWorldMatrixMap extends Map<
  VRMHumanBoneName | "hipsParent",
  THREE.Matrix4
> {}

class VRMAnimationLoaderPlugin implements GLTFLoaderPlugin {
  private readonly parser: GLTFParser;

  constructor(parser: GLTFParser, options?: VRMAnimationLoaderPluginOptions) {
    this.parser = parser;
  }

  get name(): string {
    return "VRMC_vrm_animation";
  }

  async afterRoot(gltf: GLTF): Promise<void> {
    const defGltf = gltf.parser.json as GLTFSchema.IGLTF;
    const defExtensionsUsed = defGltf.extensionsUsed;

    if (!defExtensionsUsed || !defExtensionsUsed.includes(this.name)) {
      return;
    }

    const defExtension = defGltf.extensions?.[this.name] as VRMCVRMAnimation;
    if (!defExtension) {
      return;
    }

    const nodeMap = this._createNodeMap(defExtension);
    const worldMatrixMap = await this._createBoneWorldMatrixMap(gltf, defExtension);

    const hipsParentWorldMatrix = worldMatrixMap.get("hipsParent") || MAT4_IDENTITY;
    const hipsNode = defExtension.humanoid.humanBones["hips"]!.node;
    const hips = await gltf.parser.getDependency("node", hipsNode) as THREE.Object3D;
    const restHipsPosition = hips.getWorldPosition(_v3A).applyMatrix4(hipsParentWorldMatrix);

    const animations = await this._parseAnimations(gltf.animations, defGltf.animations!, nodeMap, worldMatrixMap, restHipsPosition);

    gltf.userData.vrmAnimations = animations;
  }

  private _createNodeMap(defExtension: VRMCVRMAnimation): IVRMAnimationLoaderPluginNodeMap {
    const humanoidIndexToName = new Map<number, VRMHumanBoneName>();
    const expressionsIndexToName = new Map<number, string>();

    // humanoid
    const humanBones = defExtension.humanoid?.humanBones;
    if (humanBones) {
      Object.entries(humanBones).forEach(([name, bone]) => {
        const { node } = bone;
        if (node != null) {
          humanoidIndexToName.set(node, name as VRMHumanBoneName);
        }
      });
    }

    // expressions
    const preset = defExtension.expressions?.preset;
    const custom = defExtension.expressions?.custom;

    [...(preset || []), ...(custom || [])].forEach((expression, index) => {
      const { node } = expression;
      if (node != null) {
        expressionsIndexToName.set(node, `${index}`);
      }
    });

    // lookAt
    const lookAtIndex = defExtension.lookAt?.node ?? null;

    return { humanoidIndexToName, expressionsIndexToName, lookAtIndex };
  }

  private async _createBoneWorldMatrixMap(gltf: GLTF, defExtension: VRMCVRMAnimation): Promise<IVRMAnimationLoaderPluginWorldMatrixMap> {
    await gltf.scene.updateMatrixWorld(false);

    const threeNodes = await gltf.parser.getDependencies("node") as THREE.Object3D[];

    const worldMatrixMap: IVRMAnimationLoaderPluginWorldMatrixMap = new Map();
    Object.entries(defExtension.humanoid.humanBones).forEach(([boneName, { node }]) => {
      const threeNode = threeNodes[node];
      worldMatrixMap.set(boneName as VRMHumanBoneName, threeNode.matrixWorld);
      if (boneName === "hips") {
        worldMatrixMap.set("hipsParent", threeNode.parent?.matrixWorld || MAT4_IDENTITY);
      }
    });

    return worldMatrixMap;
  }

  private async _parseAnimations(animationClips: THREE.AnimationClip[], defAnimations: GLTFSchema.IAnimation[], nodeMap: IVRMAnimationLoaderPluginNodeMap, worldMatrixMap: IVRMAnimationLoaderPluginWorldMatrixMap, restHipsPosition: THREE.Vector3): Promise<VRMAnimation[]> {
    const animations: VRMAnimation[] = [];
    for (let iAnimation = 0; iAnimation < animationClips.length; iAnimation++) {
      const animationClip = animationClips[iAnimation];
      const defAnimation = defAnimations[iAnimation];

      const animation = new VRMAnimation();
      animation.duration = animationClip.duration;

      defAnimation.channels.forEach((channel, iChannel) => {
        const { node, path } = channel.target;
        const origTrack = animationClip.tracks[iChannel];

        if (node == null) {
          return;
        }

        // humanoid
        const boneName = nodeMap.humanoidIndexToName.get(node);
        if (boneName) {
          const parentBoneName = this._findParentBoneName(boneName, worldMatrixMap);

          if (path === "translation") {
            const track = this._transformTrackToHipsParentCoordinates(origTrack, parentBoneName, worldMatrixMap);
            animation.humanoidTracks.translation.set(boneName, track);
          } else if (path === "rotation") {
            const track = this._transformTrackToHipsParentCoordinates(origTrack, parentBoneName, worldMatrixMap);
            animation.humanoidTracks.rotation.set(boneName, track);
          } else {
            throw new Error(`Invalid path "${path}"`);
          }
        }

        // expressions
        const expressionName = nodeMap.expressionsIndexToName.get(node);
        if (expressionName) {
          if (path === "translation") {
            const times = origTrack.times;
            const values = new Float32Array(origTrack.values.length / 3);
            for (let i = 0; i < values.length; i++) {
              values[i] = origTrack.values[3 * i];
            }
            const newTrack = new THREE.NumberKeyframeTrack(`${expressionName}.weight`, times as any, values as any);
            animation.expressionTracks.set(expressionName, newTrack);
          } else {
            throw new Error(`Invalid path "${path}"`);
          }
        }

        // lookAt
        if (node === nodeMap.lookAtIndex) {
          if (path === "rotation") {
            animation.lookAtTrack = origTrack;
          } else {
            throw new Error(`Invalid path "${path}"`);
          }
        }
      });

      animation.restHipsPosition = restHipsPosition.clone();
      animations.push(animation);
    }
    return animations;
  }

  private _findParentBoneName(boneName: VRMHumanBoneName, worldMatrixMap: IVRMAnimationLoaderPluginWorldMatrixMap): VRMHumanBoneName | "hipsParent" {
    let parentBoneName: VRMHumanBoneName | "hipsParent" | null = VRMHumanBoneParentMap[boneName];
    while (parentBoneName && !worldMatrixMap.has(parentBoneName)) {
      parentBoneName = VRMHumanBoneParentMap[parentBoneName];
    }
    return parentBoneName || "hipsParent";
  }

  private _transformTrackToHipsParentCoordinates(track: THREE.KeyframeTrack, parentBoneName: VRMHumanBoneName | "hipsParent", worldMatrixMap: IVRMAnimationLoaderPluginWorldMatrixMap): THREE.KeyframeTrack {
    const worldMatrix = worldMatrixMap.get(parentBoneName)!;
    const trackValues = arrayChunk(track.values, 3).flatMap(v => _v3A.fromArray(v).applyMatrix4(worldMatrix).toArray());
    const newTrack = track.clone();
    newTrack.values = new Float32Array(trackValues);
    return newTrack;
  }
}

export { VRMAnimationLoaderPlugin };