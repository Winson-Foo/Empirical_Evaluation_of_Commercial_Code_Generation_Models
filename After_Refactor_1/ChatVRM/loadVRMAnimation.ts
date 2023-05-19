import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { VRMAnimation } from './VRMAnimation';
import { VRMAnimationLoaderPlugin } from './VRMAnimationLoaderPlugin';

const gltfLoader = new GLTFLoader();
gltfLoader.register((parser) => new VRMAnimationLoaderPlugin(parser));

/**
 * Loads a VRM animation from the given URL.
 * @param url - The URL of the VRM animation.
 * @returns A Promise that resolves with the loaded VRM animation, or null if it fails.
 */
export async function loadVRMAnimation(url: string): Promise<VRMAnimation | null> {
  try {
    // Load the VRM animation with the GLTF loader
    const gltf = await gltfLoader.loadAsync(url);

    // Get the VRM animations from the gltf userData
    const vrmAnimations: VRMAnimation[] = gltf.userData.vrmAnimations;

    // If at least one VRM animation is found, return the first animation.
    // Otherwise, return null.
    return vrmAnimations.length > 0 ? vrmAnimations[0] : null;
  } catch (error) {
    // Log any errors that occur while loading the VRM animation
    console.error(`Failed to load VRM animation at URL: ${url}`, error);
    return null;
  }
}