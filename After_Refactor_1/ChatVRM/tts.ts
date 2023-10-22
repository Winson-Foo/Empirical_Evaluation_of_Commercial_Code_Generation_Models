import fetch from "node-fetch";
import { synthesizeVoice } from "@/features/koeiromap/koeiromap";
import type { NextApiRequest, NextApiResponse } from "next";

type Data = {
  audio: string;
};

/**
 * Handles the HTTP POST request.
 */
export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<Data>
) {
  try {
    // Extract the necessary data from the request body.
    const { message, speakerX, speakerY, style } = req.body;

    // Call the synthesizeVoice function to generate the voice.
    const voice = await synthesizeVoice(message, speakerX, speakerY, style);

    // Send the generated voice as the response.
    return res.status(200).json(voice);
  } catch (error) {
    // Handle any errors that may occur.
    console.error("Error generating voice:", error);
    return res.status(500).json({ error: "Internal server error" });
  }
}

