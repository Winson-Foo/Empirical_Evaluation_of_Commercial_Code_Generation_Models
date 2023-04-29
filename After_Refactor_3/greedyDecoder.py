class GreedyDecoder(Decoder):
    def __init__(self, labels: List[str], blank_index: int = 0) -> None:
        super(GreedyDecoder, self).__init__(labels, blank_index)

    def convert_to_strings(self,
                           sequences: torch.Tensor,
                           sizes: torch.Tensor = None,
                           remove_repetitions: bool = False,
                           return_offsets: bool = False) -> List[List[str]]:
        """Given a list of numeric sequences, returns the corresponding strings"""
        strings = []
        offsets = [] if return_offsets else None
        for x in range(len(sequences)):
            seq_len = sizes[x] if sizes is not None else len(sequences[x])
            string, string_offsets = self.process_string(sequences[x], seq_len, remove_repetitions)
            strings.append([string])  # We only return one path
            if return_offsets:
                offsets.append([string_offsets])
        if return_offsets:
            return strings, offsets
        else:
            return strings

    def process_string(self,
                       sequence: torch.Tensor,
                       size: int,
                       remove_repetitions: bool = False) -> List:
        string = ''
        offsets = []
        for i in range(size):
            char = self.int_to_char[sequence[i].item()]
            if char != self.int_to_char[self.blank_index]:
                # if this char is a repetition and remove_repetitions=true, then skip
                if remove_repetitions and i != 0 and char == self.int_to_char[sequence[i - 1].item()]:
                    pass
                elif char == self.labels[self.space_index]:
                    string += ' '
                    offsets.append(i)
                else:
                    string = string + char
                    offsets.append(i)
        return string, torch.tensor(offsets, dtype=torch.int)

    def decode(self, probs: torch.Tensor, sizes: torch.Tensor = None) -> List[List[str]]:
        """
        Returns the argmax decoding given the probability matrix. Removes
        repeated elements in the sequence, as well as blanks.

        Args:
            probs: Tensor of character probabilities from the network. Expected shape of batch x seq_length x output_dim
            sizes: Size of each sequence in the mini-batch
        Returns:
            List[List[str]]: sequences of the model's best guess for the transcription on inputs
        """
        _, max_probs = torch.max(probs, 2)
        strings, offsets = self.convert_to_strings(max_probs.view(max_probs.size(0), max_probs.size(1)),
                                                   sizes,
                                                   remove_repetitions=True,
                                                   return_offsets=True)
        return strings, offsets