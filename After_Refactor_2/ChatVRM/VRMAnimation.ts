import * as THREE from 'three';
import { VRM, VRMExpressionManager, VRMHumanBoneName } from '@pixiv/three-vrm';

type HumanoidTracks = {
  translation: Map<VRMHumanBoneName, THREE.VectorKeyframeTrack>;
  rotation: Map<VRMHumanBoneName, THREE.VectorKeyframeTrack>;
};

export class VRMAnimation {
  duration: number;
  restHipsPosition: THREE.Vector3;
  humanoidTracks: HumanoidTracks;
  expressionTracks: Map<string, THREE.NumberKeyframeTrack>;
  lookAtTrack: THREE.QuaternionKeyframeTrack | null;

  constructor() {
    this.duration = 0.0;
    this.restHipsPosition = new THREE.Vector3();
    this.humanoidTracks = { translation: new Map(), rotation: new Map() };
    this.expressionTracks = new Map();
    this.lookAtTrack = null;
  }

  createAnimationClip(vrm: VRM): THREE.AnimationClip {
    const { humanoidTracks, duration, expressionTracks, lookAtTrack } = this;
    const tracks: THREE.KeyframeTrack[] = [
      ...this.createHumanoidTracks(vrm),
      ...(vrm.expressionManager ? this.createExpressionTracks(vrm.expressionManager) : []),
      ...(vrm.lookAt && lookAtTrack ? [this.createLookAtTrack('lookAtTargetParent.quaternion')] : [])
    ];

    return new THREE.AnimationClip('Clip', duration, tracks);
  }

  private createHumanoidTracks(vrm: VRM): THREE.KeyframeTrack[] {
    const { humanoidTracks } = this;
    const metaVersion = vrm.meta.metaVersion;

    return [
      ...createHumanoidTracks(humanoidTracks.rotation, vrm.humanoid, 'quaternion', (v, i) =>
        metaVersion === '0' && i % 2 === 0 ? -v : v
      ),
      ...createScaledHumanoidTracks(
        humanoidTracks.translation,
        vrm.humanoid,
        'position',
        (v, i) => (metaVersion === '0' && i % 3 !== 1 ? -v : v)
      )
    ];
  }

  private createExpressionTracks(expressionManager: VRMExpressionManager): THREE.KeyframeTrack[] {
    const { expressionTracks } = this;

    return Array.from(expressionTracks).reduce((tracks, [name, origTrack]) => {
      const trackName = expressionManager.getExpressionTrackName(name);
      if (trackName) {
        const track = origTrack.clone();
        track.name = trackName;
        return [...tracks, track];
      }
      return tracks;
    }, [] as THREE.KeyframeTrack[]);
  }

  private createLookAtTrack(trackName: string): THREE.KeyframeTrack | null {
    const { lookAtTrack } = this;

    if (!lookAtTrack) {
      return null;
    }

    const track = lookAtTrack.clone();
    track.name = trackName;
    return track;
  }
}

function createHumanoidTracks(
  tracks: HumanoidTracks['rotation'],
  humanoid: VRM['humanoid'],
  trackName: string,
  mapValue: (v: number, i: number) => number
): THREE.VectorKeyframeTrack[] {
  return Array.from(tracks).reduce((result, [name, origTrack]) => {
    const nodeName = humanoid.getNormalizedBoneNode(name)?.name;
    if (nodeName) {
      const track = new THREE.VectorKeyframeTrack(
        `${nodeName}.${trackName}`,
        origTrack.times,
        origTrack.values.map(mapValue)
      );
      return [...result, track];
    }
    return result;
  }, [] as THREE.VectorKeyframeTrack[]);
}

function createScaledHumanoidTracks(
  tracks: HumanoidTracks['translation'],
  humanoid: VRM['humanoid'],
  trackName: string,
  mapValue: (v: number, i: number) => number
): THREE.VectorKeyframeTrack[] {
  const animationY = tracks.values().next().value.values[1];
  const humanoidY = humanoid.getNormalizedAbsolutePose().hips!.position![1];
  const scale = humanoidY / animationY;

  return Array.from(tracks).reduce((result, [name, origTrack]) => {
    const nodeName = humanoid.getNormalizedBoneNode(name)?.name;
    if (nodeName) {
      const track = origTrack.clone();
      track.name = `${nodeName}.${trackName}`;
      track.values = track.values.map(mapValue).map((v) => v * scale);
      return [...result, track];
    }
    return result;
  }, [] as THREE.VectorKeyframeTrack[]);
}

