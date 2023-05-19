import { wait } from "@/utils/wait";
import { synthesizeVoice } from "../koeiromap/koeiromap";
import { Viewer } from "../vrmViewer/viewer";
import { Screenplay } from "./messages";
import { Talk } from "./messages";

type AudioBuffer = ArrayBuffer;

const fetchAudio = async (talk: Talk): Promise<AudioBuffer | null> => {
  try {
    const ttsVoice = await synthesizeVoice(
      talk.message,
      talk.speakerX,
      talk.speakerY,
      talk.style
    );
    const url = ttsVoice.audio;

    if (url == null) {
      return null;
    }

    const resAudio = await fetch(url);
    const buffer = await resAudio.arrayBuffer();
    return buffer;
  } catch (err) {
    console.error(err);
    return null;
  }
};

const speakAudio = async (
  viewer: Viewer,
  audioBuffer: AudioBuffer,
  screenplay: Screenplay,
  onStart?: () => void,
  onComplete?: () => void
): Promise<void> => {
  try {
    onStart?.();
    if (viewer.model != null) {
      await viewer.model.speak(audioBuffer, screenplay);
    }
  } catch (err) {
    console.error(err);
  } finally {
    onComplete?.();
  }
};

const createCharacterSpeaker = () => {
  return async (
    screenplay: Screenplay,
    viewer: Viewer,
    onStart?: () => void,
    onComplete?: () => void
  ): Promise<void> => {
    const buffer = await fetchAudio(screenplay.talk);
    if (buffer == null) {
      return;
    }

    const now = Date.now();
    await wait(Math.max(0, 1000 - (now - lastTime)));
    lastTime = Date.now();
    await speakAudio(viewer, buffer, screenplay, onStart, onComplete);
  };
};

export const speakCharacter = createCharacterSpeaker();