import { wait } from "@/utils/wait";
import { synthesizeVoice } from "../koeiromap/koeiromap";
import { Viewer } from "../vrmViewer/viewer";
import { Screenplay } from "./messages";
import { Talk } from "./messages";

interface TalkOptions {
  message: string;
  speakerX: number;
  speakerY: number;
  style: string;
}

interface SpeakCharacterOptions {
  onStart?: () => void;
  onComplete?: () => void;
}

export const fetchAudio = async ({ message, speakerX, speakerY, style }: TalkOptions): Promise<ArrayBuffer> => {
  const ttsVoice = await synthesizeVoice(message, speakerX, speakerY, style);
  const url = ttsVoice?.audio;

  if (!url) {
    throw new Error("Something went wrong during audio fetch.");
  }

  const resAudio = await fetch(url);
  return resAudio.arrayBuffer();
};

export const speakCharacter = async (screenplay: Screenplay, viewer: Viewer, options: SpeakCharacterOptions = {}): Promise<void> => {
  const { onStart, onComplete } = options;

  const now = Date.now();
  await wait(Math.max(lastTime + 1000 - now, 0));
  lastTime = Date.now();

  const audioBuffer = await fetchAudio(screenplay.talk).catch(() => null);
  if (audioBuffer) {
    onStart?.();
    await viewer.model?.speak(audioBuffer, screenplay);
  }

  onComplete?.();
};

let lastTime = 0;