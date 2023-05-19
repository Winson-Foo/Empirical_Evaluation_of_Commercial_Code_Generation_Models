import fetch from "node-fetch";
import { synthesizeVoice } from "@/features/koeiromap/koeiromap";
import type { NextApiRequest, NextApiResponse } from "next";

type VoiceRequest = {
  message: string;
  speakerX: string;
  speakerY: string;
  style: string;
};

type VoiceResponse = {
  audio: string;
};

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<VoiceResponse>
) {
  try {
    const { message, speakerX, speakerY, style }: VoiceRequest = req.body;
    const synthesizedVoice: string = await synthesizeVoice(message, speakerX, speakerY, style);
    res.status(200).json({ audio: synthesizedVoice });
  } catch (error) {
    res.status(500).json({ message: "Error processing voice request" });
  }
}

