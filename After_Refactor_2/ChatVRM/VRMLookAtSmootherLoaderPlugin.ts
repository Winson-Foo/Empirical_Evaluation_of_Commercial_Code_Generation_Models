import {
  VRMHumanoid,
  VRMLookAt,
  VRMLookAtLoaderPlugin,
} from "@pixiv/three-vrm";
import { GLTF } from "three/examples/jsm/loaders/GLTFLoader";
import { VRMLookAtSmoother } from "./VRMLookAtSmoother";

export class VRMLookAtSmootherLoaderPlugin extends VRMLookAtLoaderPlugin {
  public readonly name = "VRMLookAtSmootherLoaderPlugin";

  public async afterRoot(gltf: GLTF): Promise<void> {
    await super.afterRoot(gltf);

    const humanoid = this.getHumanoid(gltf);
    const lookAt = this.getLookAt(gltf);

    if (humanoid && lookAt) {
      const lookAtSmoother = new VRMLookAtSmoother(humanoid, lookAt.applier);
      lookAtSmoother.copy(lookAt);
      gltf.userData.vrmLookAt = lookAtSmoother;
    }
  }

  private getHumanoid(gltf: GLTF): VRMHumanoid | null {
    return gltf.userData.vrmHumanoid ?? null;
  }

  private getLookAt(gltf: GLTF): VRMLookAt | null {
    return gltf.userData.vrmLookAt ?? null;
  }
}

// Improvements:
// 1. Added readonly to name property to prevent accidental reassignment
// 2. Extracted getHumanoid and getLookAt methods for better readability and maintainability
// 3. Simplified if statement by using truthy values instead of explicitly checking for null
// 4. Used optional chaining for userData properties for safer and more concise code.

