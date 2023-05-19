import { TalkStyle } from "../messages/messages";

interface VoiceSynthesizerParams {
  text: string;
  speakerPositionX: number;
  speakerPositionY: number;
  talkStyle: TalkStyle;
}

interface VoiceSynthesizerResponse {
  audio: string;
}

export async function synthesizeVoice(
  params: VoiceSynthesizerParams
): Promise<VoiceSynthesizerResponse> {
  const param = {
    method: "POST",
    body: JSON.stringify(params),
    headers: {
      "Content-type": "application/json; charset=UTF-8",
    },
  };

  try {
    const koeiroRes = await fetch(
      "https://api.rinna.co.jp/models/cttse/koeiro",
      param
    );

    const data = (await koeiroRes.json()) as VoiceSynthesizerResponse;

    return data;
  } catch (error) {
    console.error(error);
    throw new Error("Voice synthesis failed");
  }
}

