import * as THREE from "three";
import { VRM, VRMExpressionManager, VRMHumanBoneName } from "@pixiv/three-vrm";

interface HumanoidTracks {
  translation: Map<VRMHumanBoneName, THREE.VectorKeyframeTrack>;
  rotation: Map<VRMHumanBoneName, THREE.VectorKeyframeTrack>;
}

interface ExpressionTracks extends Map<string, THREE.NumberKeyframeTrack> {}

export class VRMAnimation {
  duration: number;
  restHipsPosition: THREE.Vector3;
  humanoidTracks: HumanoidTracks;
  expressionTracks: ExpressionTracks;
  lookAtTrack: THREE.QuaternionKeyframeTrack | null;

  constructor() {
    this.duration = 0.0;
    this.restHipsPosition = new THREE.Vector3();
    this.humanoidTracks = {
      translation: new Map(),
      rotation: new Map(),
    };
    this.expressionTracks = new Map();
    this.lookAtTrack = null;
  }

  createVRMAnimationClip(vrm: VRM): THREE.AnimationClip {
    const tracks: THREE.KeyframeTrack[] = [];

    tracks.push(...this.createVRMHumanoidTracks(vrm));

    if (vrm.expressionManager) {
      tracks.push(...this.createVRMExpressionTracks(vrm.expressionManager));
    }

    if (vrm.lookAt) {
      const track = this.createLookAtQuaternionTrack("lookAtTargetParent.quaternion");
      if (track) {
        tracks.push(track);
      }
    }

    return new THREE.AnimationClip("Clip", this.duration, tracks);
  }

  createVRMHumanoidTracks(vrm: VRM): THREE.KeyframeTrack[] {
    const humanoid = vrm.humanoid;
    const metaVersion = vrm.meta.metaVersion;
    const tracks: THREE.KeyframeTrack[] = [];

    tracks.push(...this.createHumanoidRotationTracks(humanoid, metaVersion));
    tracks.push(...this.createHumanoidTranslationTracks(humanoid, metaVersion));

    return tracks;
  }

  createHumanoidRotationTracks(humanoid: any, metaVersion: string): THREE.KeyframeTrack[] {
    const tracks: THREE.VectorKeyframeTrack[] = [];

    for (const [name, origTrack] of this.humanoidTracks.rotation.entries()) {
      const nodeName = humanoid.getNormalizedBoneNode(name)?.name;
      if (nodeName) {
        const track = new THREE.VectorKeyframeTrack(
          `${nodeName}.quaternion`,
          origTrack.times,
          origTrack.values.map((v, i) => (metaVersion === "0" && i % 2 === 0 ? -v : v))
        );
        tracks.push(track);
      }
    }

    return tracks;
  }

  createHumanoidTranslationTracks(humanoid: any, metaVersion: string): THREE.KeyframeTrack[] {
    const tracks: THREE.VectorKeyframeTrack[] = [];

    for (const [name, origTrack] of this.humanoidTracks.translation.entries()) {
      const nodeName = humanoid.getNormalizedBoneNode(name)?.name;
      if (nodeName) {
        const animationY = this.restHipsPosition.y;
        const humanoidY = humanoid.getNormalizedAbsolutePose().hips!.position![1];
        const scale = humanoidY / animationY;

        const track = origTrack.clone();
        track.values = track.values.map(
          (v, i) => (metaVersion === "0" && i % 3 !== 1 ? -v : v) * scale
        );
        track.name = `${nodeName}.position`;
        tracks.push(track);
      }
    }

    return tracks;
  }

  createVRMExpressionTracks(expressionManager: VRMExpressionManager): THREE.KeyframeTrack[] {
    const tracks: THREE.KeyframeTrack[] = [];

    for (const [name, origTrack] of this.expressionTracks.entries()) {
      const trackName = expressionManager.getExpressionTrackName(name);
      if (trackName) {
        const track = origTrack.clone();
        track.name = trackName;
        tracks.push(track);
      }
    }

    return tracks;
  }

  createLookAtQuaternionTrack(trackName: string): THREE.QuaternionKeyframeTrack | null {
    if (!this.lookAtTrack) {
      return null;
    }

    const track = this.lookAtTrack.clone();
    track.name = trackName;
    return track;
  }
}

