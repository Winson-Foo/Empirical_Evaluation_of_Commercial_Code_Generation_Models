import { wait } from "@/utils/wait";
import { synthesizeVoice } from "../koeiromap/koeiromap";
import { Viewer } from "../vrmViewer/viewer";
import { Screenplay } from "./messages";
import { Talk } from "./messages";

const createSpeakCharacter = () => {
  let lastTime = 0;

  return async (
    screenplay: Screenplay,
    viewer: Viewer
  ) => {
    const now = Date.now();
    if (now - lastTime < 1000) {
      await wait(1000 - (now - lastTime));
    }

    const audioBuffer = await fetchAudio(screenplay.talk);

    lastTime = Date.now();
    viewer.model?.speak(audioBuffer, screenplay);
  };
};

export const speakCharacter = createSpeakCharacter();

export const fetchAudio = async (talk: Talk): Promise<ArrayBuffer> => {
  const ttsVoice = await synthesizeVoice(
    talk.message,
    talk.speakerX,
    talk.speakerY,
    talk.style
  );
  const url = ttsVoice.audio;

  if (url == null) {
    throw new Error("Something went wrong");
  }

  const resAudio = await fetch(url);
  const buffer = await resAudio.arrayBuffer();
  return buffer;
};