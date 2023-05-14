import { TalkStyle } from "../messages/messages";

interface VoiceData {
  audio: string;
}

interface ApiConfig {
  endpoint: string;
  headers: Headers;
}

const apiConfig: ApiConfig = {
  endpoint: "https://api.rinna.co.jp/models/cttse/koeiro",
  headers: new Headers({
    "Content-type": "application/json; charset=UTF-8",
  }),
};

export async function synthesizeVoice(
  message: string,
  speakerX: number,
  speakerY: number,
  style: TalkStyle
): Promise<VoiceData> {
  const requestData = {
    text: message,
    speaker_x: speakerX,
    speaker_y: speakerY,
    style: style,
  };

  const requestOptions = {
    method: "POST",
    headers: apiConfig.headers,
    body: JSON.stringify(requestData),
  };

  const response = await fetch(apiConfig.endpoint, requestOptions);
  const data: VoiceData = await response.json();

  return data;
}

