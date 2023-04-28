import torch
from typing import List


class Decoder:
    """
    Basic decoder class from which all other decoders inherit. Implements several
    helper functions. Subclasses should implement the decode() method.

    Args:
        labels (List[str]): mapping from integers to characters.
        blank_index (int, optional): index for the blank '_' character. Defaults to 0.
    """

    def __init__(self, labels: List[str], blank_index: int = 0) -> None:
        self.labels = labels
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
        self.blank_index = blank_index
        space_index = len(labels)
        if ' ' in labels:
            space_index = labels.index(' ')
        self.space_index = space_index

    def decode(self, probs: torch.Tensor, sizes: torch.Tensor = None) -> List[List[str]]:
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription

        Args:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            List[List[str]]: sequence of the model's best guess for the transcription
        """
        raise NotImplementedError