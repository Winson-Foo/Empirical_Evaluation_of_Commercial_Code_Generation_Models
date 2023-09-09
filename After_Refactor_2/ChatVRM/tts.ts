import fetch from "node-fetch";
import { synthesizeVoice } from "@/features/koeiromap/koeiromap";

import type { NextApiRequest, NextApiResponse } from "next";

type Data = {
  audio: string;
};

async function generateVoice(message: string, speaker_x: number, speaker_y: number, style: string): Promise<string> {
  return synthesizeVoice(message, speaker_x, speaker_y, style);
}

export default async function generateVoiceHandler(
  req: NextApiRequest,
  res: NextApiResponse<Data>
) {
  try {
    const { message, speakerX, speakerY, style } = req.body;

    if (!message) {
      throw new Error("Message is required");
    }

    const voice = await generateVoice(message, speakerX, speakerY, style);

    res.status(200).json({ audio: voice });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: error.message });
  }
} 