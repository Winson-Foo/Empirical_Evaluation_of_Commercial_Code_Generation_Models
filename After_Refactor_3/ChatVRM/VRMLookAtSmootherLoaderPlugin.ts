import { VRMHumanoid } from "@pixiv/three-vrm";
import { VRMLookAt } from "@pixiv/three-vrm";
import { VRMLookAtLoaderPlugin } from "@pixiv/three-vrm";
import { GLTF } from "three/examples/jsm/loaders/GLTFLoader";
import { VRMLookAtSmoother } from "./VRMLookAtSmoother";

// Extend the VRMLookAtLoaderPlugin class
export class VRMLookAtSmootherLoaderPlugin extends VRMLookAtLoaderPlugin {
  // Return the name of the plugin
  public get name(): string {
    return "VRMLookAtSmootherLoaderPlugin";
  }

  // Override the afterRoot() method from the parent class
  public async afterRoot(gltf: GLTF): Promise<void> {
    // Call the parent class' afterRoot() method first
    await super.afterRoot(gltf);

    // Get the humanoid and lookAt objects from userData
    const humanoid: VRMHumanoid = gltf.userData.vrmHumanoid;
    const lookAt: VRMLookAt = gltf.userData.vrmLookAt;

    // If both humanoid and lookAt are available, create a VRMLookAtSmoother object and store it in userData
    if (humanoid && lookAt) {
      const lookAtSmoother = new VRMLookAtSmoother(humanoid, lookAt.applier);
      // Copy the lookAt object to lookAtSmoother object
      lookAtSmoother.copy(lookAt);
      // Store the lookAtSmoother object in userData
      gltf.userData.vrmLookAt = lookAtSmoother;
    }
  }
}