// VRMNodeMap.ts
export interface VRMNodeMap {
  humanoidIndexToName: Record<number, VRMHumanBoneName>;
  expressionsIndexToName: Record<number, string>;
  lookAtIndex: number | null;
}

// VRMBoneWorldMatrixMap.ts
export interface VRMBoneWorldMatrixMap {
  [key: string]: THREE.Matrix4;
}

// VRMAnimationParser.ts
interface VRMAnimationChannelData {
  boneName?: VRMHumanBoneName;
  expressionName?: string;
  lookAt?: boolean;
}

export class VRMAnimationParser {
  private readonly nodeMap: VRMNodeMap;
  private readonly worldMatrixMap: VRMBoneWorldMatrixMap;

  constructor(nodeMap: VRMNodeMap, worldMatrixMap: VRMBoneWorldMatrixMap) {
    this.nodeMap = nodeMap;
    this.worldMatrixMap = worldMatrixMap;
  }

  public parseChannel(channel: GLTFSchema.IAnimationChannel): VRMAnimationChannelData {
    const { node, path } = channel.target;

    const result: VRMAnimationChannelData = {};

    const boneName = this.nodeMap.humanoidIndexToName[node];
    if (boneName != null) {
      result.boneName = boneName;
      return result;
    }

    const expressionName = this.nodeMap.expressionsIndexToName[node];
    if (expressionName != null && path === "translation") {
      result.expressionName = expressionName;
      return result;
    }

    if (node === this.nodeMap.lookAtIndex && path === "rotation") {
      result.lookAt = true;
      return result;
    }

    return result;
  }

  public parseTrack(
    channelData: VRMAnimationChannelData,
    track: THREE.KeyframeTrack
  ): VRMAnimation {
    const result = new VRMAnimation();
    result.duration = track.times[track.times.length - 1];

    if (channelData.boneName != null) {
      const parentBoneName = VRMHumanBoneParentMap[channelData.boneName] ?? "hipsParent";
      const parentWorldMatrix = this.worldMatrixMap[parentBoneName] ?? MAT4_IDENTITY;
      const boneWorldMatrix = this.worldMatrixMap[channelData.boneName] ?? MAT4_IDENTITY;

      if (track.name.endsWith("rotation")) {
        result.humanoidTracks.rotation.set(
          channelData.boneName,
          this.parseRotationTrack(track, boneWorldMatrix, parentWorldMatrix)
        );
      } else if (track.name.endsWith("translation")) {
        result.humanoidTracks.translation.set(
          channelData.boneName,
          this.parseTranslationTrack(track, parentWorldMatrix)
        );
      }
    }

    if (channelData.expressionName != null) {
      result.expressionTracks.set(
        channelData.expressionName,
        this.parseExpressionTrack(track)
      );
    }

    if (channelData.lookAt != null && channelData.lookAt === true) {
      result.lookAtTrack = track;
    }

    return result;
  }

  private parseRotationTrack(
    track: THREE.KeyframeTrack,
    worldMatrix: THREE.Matrix4,
    parentMatrix: THREE.Matrix4
  ): THREE.KeyframeTrack {
    const result = track.clone();

    const quatA = new THREE.Quaternion();
    const quatB = new THREE.Quaternion();
    const quatC = new THREE.Quaternion();

    const values = track.values as Float32Array;
    for (let i = 0; i < values.length; i += 4) {
      quatC.fromArray(values, i);

      quatA.setFromRotationMatrix(worldMatrix).normalize().invert();
      quatB.setFromRotationMatrix(parentMatrix).normalize();

      quatC.premultiply(quatB).multiply(quatA).toArray(values, i);
    }

    return result;
  }

  private parseTranslationTrack(
    track: THREE.KeyframeTrack,
    parentMatrix: THREE.Matrix4
  ): THREE.KeyframeTrack {
    const result = new THREE.NumberKeyframeTrack(
      track.name,
      track.times,
      new Float32Array(track.values.length / 3)
    );

    const values = track.values as Float32Array;
    const resultValues = result.values as Float32Array;

    const v3 = new THREE.Vector3();
    for (let i = 0; i < values.length; i += 3) {
      v3.fromArray(values, i).applyMatrix4(parentMatrix).toArray(resultValues, i / 3);
    }

    return result;
  }

  private parseExpressionTrack(track: THREE.KeyframeTrack): THREE.KeyframeTrack {
    const result = new THREE.NumberKeyframeTrack(
      track.name,
      track.times,
      new Float32Array(track.values.length / 3)
    );

    const values = track.values as Float32Array;
    const resultValues = result.values as Float32Array;

    for (let i = 0; i < values.length; i += 3) {
      resultValues[i / 3] = values[i];
    }

    return result;
  }
}

// VRMAnimationLoaderPlugin.ts
export class VRMAnimationLoaderPlugin implements GLTFLoaderPlugin {
  public readonly parser: GLTFParser;

  private readonly nodeMap: VRMNodeMap;
  private readonly worldMatrixMap: VRMBoneWorldMatrixMap;

  constructor(parser: GLTFParser, options?: VRMAnimationLoaderPluginOptions) {
    this.parser = parser;
    this.nodeMap = this.createNodeMap();
    this.worldMatrixMap = this.createBoneWorldMatrixMap();
  }

  public get name(): string {
    return "VRMC_vrm_animation";
  }

  public async afterRoot(gltf: GLTF): Promise<void> {
    const defGltf = gltf.parser.json as GLTFSchema.IGLTF;
    const defExtensionsUsed = defGltf.extensionsUsed;

    if (defExtensionsUsed == null || defExtensionsUsed.indexOf(this.name) === -1) {
      return;
    }

    const defExtension = defGltf.extensions?.[this.name] as VRMCVRMAnimation | undefined;

    if (defExtension == null) {
      return;
    }

    const animationParser = new VRMAnimationParser(this.nodeMap, this.worldMatrixMap);

    const animations = await Promise.all(
      gltf.animations.map(async (clip) => {
        const defAnimation = defGltf.animations![gltf.animations.indexOf(clip)];
        const channelData = defAnimation.channels.map((channel) =>
          animationParser.parseChannel(channel)
        );
        const parsedTracks = clip.tracks.map((track) =>
          animationParser.parseTrack(
            channelData[defAnimation.channels.indexOf(track.name)],
            track
          )
        );
        const result = parsedTracks.reduce((acc, val) => acc.merge(val));
        result.restHipsPosition = this.getRestHipsPosition(defExtension);
        return result;
      })
    );

    gltf.userData.vrmAnimations = animations;
  }

  private createNodeMap(): VRMNodeMap {
    const defExtension = this.parser.json.extensions?.[this.name] as VRMCVRMAnimation | undefined;

    const humanoidIndexToName: Record<number, VRMHumanBoneName> = {};
    const expressionsIndexToName: Record<number, string> = {};

    if (defExtension?.humanoid?.humanBones) {
      Object.entries(defExtension.humanoid.humanBones).forEach(([name, { node }]) => {
        humanoidIndexToName[node] = name as VRMHumanBoneName;
      });
    }

    if (defExtension?.expressions) {
      Object.entries(defExtension.expressions).forEach(([category, expressions]) => {
        Object.entries(expressions).forEach(([name, { node }]) => {
          expressionsIndexToName[node] = `${category}.${name}`;
        });
      });
    }

    const lookAtIndex = defExtension?.lookAt?.node ?? null;

    return { humanoidIndexToName, expressionsIndexToName, lookAtIndex };
  }

  private async createBoneWorldMatrixMap(): Promise<VRMBoneWorldMatrixMap> {
    const nodes = await this.parser.getDependencies("node");
    const gltf = this.parser.json as GLTFSchema.IGLTF;
    const bones = gltf.extensions?.[this.name]?.humanoid?.humanBones;

    const worldMatrixMap: VRMBoneWorldMatrixMap = {};

    if (bones) {
      Object.entries(bones).forEach(([boneName, { node }]) => {
        const nodeIndex = parseInt(node);
        const nodeObject = nodes[nodeIndex];
        worldMatrixMap[boneName] = nodeObject.matrixWorld;
      });

      const hipsParent = bones.hips?.parent
        ? nodes[parseInt(bones.hips.parent)].matrixWorld
        : MAT4_IDENTITY;
      worldMatrixMap["hipsParent"] = hipsParent;
    }

    return worldMatrixMap;
  }

  private getRestHipsPosition(defExtension

