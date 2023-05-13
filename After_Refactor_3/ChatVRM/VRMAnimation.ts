import * as THREE from "three";
import { VRM, VRMExpressionManager, VRMHumanBoneName } from "@pixiv/three-vrm";

export class VRMAnimation {
  // duration and initial rest hips position are public properties
  public durationInSeconds: number;
  public initialRestHipsPosition: THREE.Vector3;

  // humanoid tracks and expression tracks are private properties
  private humanoidRotationTracks: Map<VRMHumanBoneName, THREE.VectorKeyframeTrack>;
  private humanoidTranslationTracks: Map<VRMHumanBoneName, THREE.VectorKeyframeTrack>;
  private expressionTracks: Map<string, THREE.NumberKeyframeTrack>;
  private lookAtTrack: THREE.QuaternionKeyframeTrack | null;

  public constructor() {
    this.durationInSeconds = 0.0;
    this.initialRestHipsPosition = new THREE.Vector3();

    this.humanoidRotationTracks = new Map();
    this.humanoidTranslationTracks = new Map();

    this.expressionTracks = new Map();
    this.lookAtTrack = null;
  }

  // create a THREE.AnimationClip from a VRM
  public createAnimationClip(vrm: VRM): THREE.AnimationClip {
    const tracks: THREE.KeyframeTrack[] = [];

    // create humanoid animation tracks
    tracks.push(...this.createHumanoidRotationTracks(vrm));
    tracks.push(...this.createHumanoidTranslationTracks(vrm));

    // create expression tracks if expression manager is available
    if (vrm.expressionManager != null) {
      tracks.push(...this.createExpressionTracks(vrm.expressionManager));
    }

    // create lookat track if lookat target is available
    if (vrm.lookAt != null) {
      const track = this.createLookAtTrack("lookAtTargetParent.quaternion");
      if (track != null) {
        tracks.push(track);
      }
    }

    // return the animation clip with all the tracks
    return new THREE.AnimationClip("VRM Animation", this.durationInSeconds, tracks);
  }

  // create rotation tracks for humanoid bones
  private createHumanoidRotationTracks(vrm: VRM): THREE.KeyframeTrack[] {
    const humanoid = vrm.humanoid;
    const metaVersion = vrm.meta.metaVersion;
    const tracks: THREE.KeyframeTrack[] = [];

    for (const [boneName, origTrack] of this.humanoidRotationTracks.entries()) {
      const normalizedBoneNode = humanoid.getNormalizedBoneNode(boneName);
      const boneNodeName = normalizedBoneNode?.name;

      if (boneNodeName != null) {
        const track = new THREE.VectorKeyframeTrack(
          `${boneNodeName}.quaternion`,
          origTrack.times,
          origTrack.values.map((value, index) =>
            // handle negative scale for older VRM versions
            metaVersion === "0" && index % 2 === 0 ? -value : value
          )
        );
        tracks.push(track);
      }
    }

    return tracks;
  }

  // create translation tracks for humanoid bones
  private createHumanoidTranslationTracks(vrm: VRM): THREE.KeyframeTrack[] {
    const humanoid = vrm.humanoid;
    const metaVersion = vrm.meta.metaVersion;
    const tracks: THREE.KeyframeTrack[] = [];

    for (const [boneName, origTrack] of this.humanoidTranslationTracks.entries()) {
      const normalizedBoneNode = humanoid.getNormalizedBoneNode(boneName);
      const boneNodeName = normalizedBoneNode?.name;

      if (boneNodeName != null) {
        // apply scale to match hip position in VRM with hip position in animation
        const animationHipPositionY = this.initialRestHipsPosition.y;
        const humanoidHipPositionY =
          humanoid.getNormalizedAbsolutePose().hips!.position![1];
        const scale = humanoidHipPositionY / animationHipPositionY;

        const track = origTrack.clone();
        track.values = track.values.map(
          (value, index) =>
            // handle negative scale for older VRM versions
            metaVersion === "0" && index % 3 !== 1 ? -value : value
        ).map(v => v * scale);
        track.name = `${boneNodeName}.position`;
        tracks.push(track);
      }
    }

    return tracks;
  }

  // create expression tracks from expression manager
  private createExpressionTracks(
    expressionManager: VRMExpressionManager
  ): THREE.KeyframeTrack[] {
    const tracks: THREE.KeyframeTrack[] = [];

    for (const [expressionName, origTrack] of this.expressionTracks.entries()) {
      const trackName = expressionManager.getExpressionTrackName(expressionName);

      if (trackName != null) {
        const track = origTrack.clone();
        track.name = trackName;
        tracks.push(track);
      }
    }

    return tracks;
  }

  // create lookat track from lookat target
  private createLookAtTrack(trackName: string): THREE.KeyframeTrack | null {
    if (this.lookAtTrack == null) {
      return null;
    }

    const track = this.lookAtTrack.clone();
    track.name = trackName;
    return track;
  }
}

