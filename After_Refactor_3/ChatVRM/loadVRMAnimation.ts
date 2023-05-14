import { GLTFLoader, GLTF } from 'three';
import { VRMAnimation } from './VRMAnimation';

// Create a function to add VRMAnimationLoaderPlugin to the GLTFLoader
function addVRMAnimationLoaderPlugin(loader: GLTFLoader): void {
  loader.register((parser) => new VRMAnimationLoaderPlugin(parser));
}

// Create a function to extract VRMAnimation from GLTF
function extractVRMAnimation(gltf: GLTF): VRMAnimation | null {
  const vrmAnimations: VRMAnimation[] = gltf.userData.vrmAnimations;
  const vrmAnimation: VRMAnimation | undefined = vrmAnimations[0];

  return vrmAnimation ?? null;
}

// Load the VRMAnimation from the given URL
export async function loadVRMAnimation(url: string): Promise<VRMAnimation | null> {
  const loader = new GLTFLoader();
  addVRMAnimationLoaderPlugin(loader);

  try {
    const gltf = await loader.loadAsync(url);
    return extractVRMAnimation(gltf);
  } catch (error) {
    console.error(`Failed to load VRMAnimation from ${url}: ${error}`);
    return null;
  }
}