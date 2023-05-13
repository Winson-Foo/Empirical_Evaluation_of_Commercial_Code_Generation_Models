import { GLTFLoader, GLTFParser } from 'three/examples/jsm/loaders/GLTFLoader';
import { VRMAnimation } from './VRMAnimation';
import { VRMAnimationLoaderPlugin } from './VRMAnimationLoaderPlugin';

function configureLoader(loader: GLTFLoader) {
  loader.register((parser: GLTFParser) => new VRMAnimationLoaderPlugin(parser));
}

function extractVRMAnimation(gltf: any): VRMAnimation | null {
  const vrmAnimations: VRMAnimation[] = gltf.userData.vrmAnimations || [];
  const vrmAnimation: VRMAnimation | undefined = vrmAnimations[0];

  return vrmAnimation ?? null;
}

export async function loadVRMAnimationFromUrl(url: string): Promise<VRMAnimation | null> {
  const loader = new GLTFLoader();
  configureLoader(loader);

  const gltf = await loader.loadAsync(url);

  return extractVRMAnimation(gltf);
}