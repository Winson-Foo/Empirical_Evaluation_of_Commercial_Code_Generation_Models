import { TalkStyle } from "../messages/messages";

interface VoiceParams {
  text: string;
  speakerX: number;
  speakerY: number;
  style: TalkStyle;
}

export async function synthesizeVoice(params: VoiceParams) {
  try {
    const requestOptions = {
      method: "POST",
      body: JSON.stringify(params),
      headers: {
        "Content-type": "application/json; charset=UTF-8",
      },
    };

    const response = await fetch(
      "https://api.rinna.co.jp/models/cttse/koeiro",
      requestOptions
    );

    const { audio } = await response.json();

    return { audio };
  } catch (error) {
    console.error(error.message);
    // handle error
  }
}

