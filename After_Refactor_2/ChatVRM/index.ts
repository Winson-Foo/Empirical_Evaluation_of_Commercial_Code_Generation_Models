// index.ts
import { LipSyncPlayer } from "./lipSyncPlayer";

const audio = new AudioContext();
const player = new LipSyncPlayer(audio);

player.playFromURL("https://example.com/audio.mp3").then(analyzer => {
  console.log(analyzer.analyze());
});