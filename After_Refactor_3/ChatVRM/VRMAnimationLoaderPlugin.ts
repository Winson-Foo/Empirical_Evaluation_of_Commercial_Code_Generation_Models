import * as THREE from "three";
import { GLTF, GLTFLoaderPlugin, GLTFParser } from "three/examples/jsm/loaders/GLTFLoader";
import { GLTF as GLTFSchema } from "@gltf-transform/core";
import { VRMAnimationLoaderPluginOptions } from "./VRMAnimationLoaderPluginOptions";
import { VRMHumanBoneName, VRMHumanBoneParentMap }�from "@pixiv/three-vrm";
import { VRMAnimation } from "./VRMAnimation";
import { arrayChunk } from "./utils/arrayChunk";

const MAT4_IDENTITY = new THREE.Matrix4();

interface NodeMap {
  humanoid: Map<number, VRMHumanBoneName>;
  expressions: Map<number, string>;
  lookAt: number | null;
}

type WorldMatrixMap = Map<VRMHumanBoneName | "hipsParent", THREE.Matrix4>;

export class VRMAnimationLoaderPlugin implements GLTFLoaderPlugin {
  public readonly parser: GLTFParser;

  public constructor(parser: GLTFParser, options?: VRMAnimationLoaderPluginOptions) {
    this.parser = parser;
  }

  public get name(): string {
    return "VRMC_vrm_animation";
  }

  public async afterRoot(gltf: GLTF): Promise<void> {
    const defGltf = gltf.parser.json as GLTFSchema.IGLTF;

    if (!defGltf.extensionsUsed?.includes(this.name)) return;

    const defExtension = defGltf.extensions?.[this.name] as VRMCVRMAnimation;

    if (!defExtension) return;

    const nodeMap = this._createNodeMap(defExtension);
    const worldMatrixMap = await this._createBoneWorldMatrixMap(gltf, defExtension);

    const hipsNode = defExtension.humanoid.humanBones["hips"]!.node;
    const hips = await gltf.parser.getDependency<THREE.Object3D>("node", hipsNode);
    const restHipsPosition = new THREE.Vector3().setFromMatrixPosition(hips.matrixWorld);

    const animations = await Promise.all(gltf.animations.map(async (clip, i) => {
      const defAnimation = defGltf.animations![i];
      return this._parseAnimation(clip, defAnimation, nodeMap, worldMatrixMap, restHipsPosition);
    }));

    gltf.userData.vrmAnimations = animations;
  }

  private _createNodeMap(extension: VRMCVRMAnimation): NodeMap {
    const humanoid: NodeMap["humanoid"] = new Map();
    const expressions: NodeMap["expressions"] = new Map();
    let lookAt: NodeMap["lookAt"] = null;

    const { humanBones, expressions: { preset, custom }, lookAt: { node } = {} } = extension;

    Object.entries(humanBones)
      .forEach(([name, bone]) => humanoid.set(bone.node, name as VRMHumanBoneName));

    (preset || custom)?.forEach(([name, { node }]) => expressions.set(node, name as string));
    lookAt = node ?? null;

    return { humanoid, expressions, lookAt };
  }

  private async _createBoneWorldMatrixMap(gltf: GLTF, extension: VRMCVRMAnimation): Promise<WorldMatrixMap> {
    await gltf.scene.traverse((node) => node.updateWorldMatrix(false, true));

    const nodes = await gltf.parser.getDependencies<THREE.Object3D>("node");
    const worldMatrixMap: WorldMatrixMap = new Map();

    for (const [boneName, { node }] of Object.entries(extension.humanoid.humanBones)) {
      const worldMatrix = nodes[node].matrixWorld;
      worldMatrixMap.set(boneName as VRMHumanBoneName, worldMatrix);

      if (boneName === "hips") {
        const hipsParentWorldMatrix = nodes[node].parent?.matrixWorld ?? MAT4_IDENTITY;
        worldMatrixMap.set("hipsParent", hipsParentWorldMatrix);
      }
    }

    return worldMatrixMap;
  }

  private _parseAnimation(
    clip: THREE.AnimationClip,
    defAnimation: GLTFSchema.IAnimation,
    nodeMap: NodeMap,
    worldMatrixMap: WorldMatrixMap,
    restHipsPosition: THREE.Vector3
  ): VRMAnimation {
    const tracks = clip.tracks;
    const channels = defAnimation.channels;

    const animation = new VRMAnimation();
    animation.duration = clip.duration;

    channels.forEach((channel, i) => {
      const { node, path } = channel.target;
      const track = tracks[i];

      if (node == null) return;

      const boneName = nodeMap.humanoid.get(node);
      if (boneName) {
        const parentBoneName = VRMHumanBoneParentMap[boneName] ?? "hipsParent";
        const worldMatrix = worldMatrixMap.get(boneName)!;
        const parentWorldMatrix = worldMatrixMap.get(parentBoneName)!;

        if (path === "translation") {
          const trackValues = arrayChunk(track.values, 3)
            .flatMap((v) => _v3A.fromArray(v).applyMatrix4(parentWorldMatrix).toArray());
          animation.humanoidTracks.translation.set(boneName, this._clonedTrack(track, trackValues));
        } else if (path === "rotation") {
          _quatA.setFromRotationMatrix(worldMatrix).normalize().invert();
          _quatB.setFromRotationMatrix(parentWorldMatrix).normalize();

          const trackValues = arrayChunk(track.values, 4)
            .flatMap((v) => _quatC.fromArray(v).premultiply(_quatB).multiply(_quatA).toArray());
          animation.humanoidTracks.rotation.set(boneName, this._clonedTrack(track, trackValues));
        } else {
          throw new Error(`Invalid path "${path}"`);
        }
      }

      const expressionName = nodeMap.expressions.get(node);
      if (expressionName) {
        if (path === "translation") {
          const times = track.times;
          const values = new Float32Array(track.values.length / 3);
          for (let i = 0; i < values.length; i++) values[i] = track.values[i * 3];

          const newTrack = new THREE.NumberKeyframeTrack(`${expressionName}.weight`, times, values);
          animation.expressionTracks.set(expressionName, newTrack);
        } else {
          throw new Error(`Invalid path "${path}"`);
        }
      }

      if (node === nodeMap.lookAt) {
        if (path === "rotation") {
          animation.lookAtTrack = track.clone();
        } else {
          throw new Error(`Invalid path "${path}"`);
        }
      }
    });

    animation.restHipsPosition = restHipsPosition;

    return animation;
  }

  private _clonedTrack(track: THREE.KeyframeTrack, values: number[]): THREE.KeyframeTrack {
    const clonedTrack = track.clone();
    clonedTrack.values = new Float32Array(values);
    return clonedTrack;
  }
}

