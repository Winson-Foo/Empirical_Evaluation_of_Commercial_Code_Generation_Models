import { VRMHumanoid, VRMLookAt, VRMLookAtLoaderPlugin } from "@pixiv/three-vrm";
import { GLTF } from "three/examples/jsm/loaders/GLTFLoader";
import { VRMLookAtSmoother } from "./VRMLookAtSmoother";

export class VRMLookAtSmootherLoaderPlugin extends VRMLookAtLoaderPlugin {
  public get name(): string {
    return "VRMLookAtSmootherLoaderPlugin";
  }

  public async afterRoot(gltf: GLTF): Promise<void> {
    await super.afterRoot(gltf);
    try {
      const {vrmHumanoid as humanoid, vrmLookAt as lookAt} = gltf.userData;
      if (!humanoid || !lookAt) {
        throw new Error('UserData not found');
      }
      const lookAtSmoother = new VRMLookAtSmoother(humanoid, lookAt.applier);
      lookAtSmoother.copy(lookAt);
      gltf.userData.vrmLookAt = lookAtSmoother;
    } catch (error) {
      console.error(`Error in VRMLookAtSmootherLoaderPlugin: ${error}`);
    }
  }
} 