class BeamCTCDecoder(Decoder):
    def __init__(self,
                 labels: List[str],
                 lm_path: str = None,
                 alpha: float = 0,
                 beta: float = 0,
                 cutoff_top_n: int = 40,
                 cutoff_prob: float = 1.0,
                 beam_width: int = 100,
                 num_processes: int = 4,
                 blank_index: int = 0) -> None:
        super(BeamCTCDecoder, self).__init__(labels, blank_index)
        try:
            from ctcdecode import CTCBeamDecoder
        except ImportError:
            raise ImportError("BeamCTCDecoder requires paddledecoder package.")
        self._decoder = CTCBeamDecoder(
            list(labels), lm_path, alpha, beta, cutoff_top_n, cutoff_prob, beam_width, num_processes, blank_index)

    def convert_to_strings(self, out: torch.Tensor, seq_len: torch.Tensor) -> List[List[str]]:
        results = []
        for b, batch in enumerate(out):
            utterances = []
            for p, utt in enumerate(batch):
                size = seq_len[b][p]
                if size > 0:
                    transcript = ''.join([self.int_to_char[x.item()] for x in utt[0:size]])
                else:
                    transcript = ''
                utterances.append(transcript)
            results.append(utterances)
        return results

    def convert_tensor(self, offsets: List[List[int]], sizes: torch.Tensor) -> List[List[torch.Tensor]]:
        results = []
        for b, batch in enumerate(offsets):
            utterances = []
            for p, utt in enumerate(batch):
                size = sizes[b][p]
                if sizes[b][p] > 0:
                    utterances.append(utt[0:size])
                else:
                    utterances.append(torch.tensor([], dtype=torch.int))
            results.append(utterances)
        return results

    def decode(self, probs: torch.Tensor, sizes: torch.Tensor = None) -> List[List[str]]:
        """
        Decodes probability output using ctcdecode package.

        Args:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes: Size of each sequence in the mini-batch
        Returns:
            List[List[str]]: sequences of the model's best guess for the transcription
        """
        probs = probs.cpu()
        out, scores, offsets, seq_lens = self._decoder.decode(probs, sizes)

        strings = self.convert_to_strings(out, seq_lens)
        offsets = self.convert_tensor(offsets, seq_lens)
        return strings, offsets