#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
# Modified to support PyTorch Tensors

import torch
from ctcdecode import CTCBeamDecoder


class Decoder:
    """
    A basic decoder that implements helper functions for all other decoders.
    Subclasses should implement the decode() method.

    Args:
        labels (list): A list of characters that maps to integers.
        blank_index (int, optional): The index of the blank '_' character. Defaults to 0.
    """
    def __init__(self, labels, blank_index=0):
        self.labels = labels
        self.int_to_char = {i: c for i, c in enumerate(labels)}
        self.blank_index = blank_index
        if ' ' in labels:
            space_index = labels.index(' ')
        else:
            space_index = len(labels)
        self.space_index = space_index

    def decode(self, probs, sizes=None):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription.

        Args:
            probs: A tensor of character probabilities where probs[c, t]
                is the probability of character c at time t.
            sizes (optional): A list of the size of each sequence in the mini-batch.

        Returns:
            A list of sequences of the model's best guess for the transcription.
        """
        raise NotImplementedError


class BeamCTCDecoder(Decoder):
    """
    A CTC beam decoder.

    Args:
        labels (list): A list of characters that maps to integers.
        lm_path (str, optional): The path to the language model.
        alpha (float, optional): The weight for the language model.
        beta (float, optional): The weight for the word count.
        cutoff_top_n (int, optional): The number of beams to keep.
        cutoff_prob (float, optional): The minimum probability for a beam to be kept.
        beam_width (int, optional): The number of beams to search for each time-step.
        num_processes (int, optional): The number of processes to use for beam decoding.
        blank_index (int, optional): The index of the blank '_' character. Defaults to 0.
    """
    def __init__(
        self, labels, lm_path=None, alpha=0, beta=0, cutoff_top_n=40,
        cutoff_prob=1.0, beam_width=100, num_processes=4, blank_index=0,
    ):
        super().__init__(labels, blank_index)
        self.decoder = CTCBeamDecoder(
            labels, lm_path, alpha, beta, cutoff_top_n, cutoff_prob,
            beam_width, num_processes, blank_index,
        )

    def convert_to_strings(self, out, seq_len):
        """Converts the output of decode() to a list of strings."""
        results = []
        for i, tensor in enumerate(out):
            utts = [
                ''.join(self.int_to_char[c.item()] for c in tensor[j, :seq_len[i][j]])
                for j in range(tensor.size(0))
            ]
            results.append(utts)
        return results

    def decode(self, probs, sizes=None):
        """
        Decodes probability output using ctcdecode (CTC Beam Search Decoding) package.

        Args:
            probs (tensor): A tensor of character probabilities where probs[c, t]
                is the probability of character c at time t.
            sizes (optional): A list of the size of each sequence in the mini-batch.

        Returns:
            A list of sequences of the model's best guess for the transcription.
        """
        probs_cpu = probs.cpu()
        out, scores, offsets, seq_lens = self.decoder.decode(probs_cpu, sizes)
        strings = self.convert_to_strings(out, seq_lens)
        return strings


class GreedyDecoder(Decoder):
    """
    A decoder that returns the argmax decoding given the probability matrix.

    Args:
        labels (list): A list of characters that maps to integers.
        blank_index (int, optional): The index of the blank '_' character. Defaults to 0.
    """
    def __init__(self, labels, blank_index=0):
        super().__init__(labels, blank_index)

    def convert_to_strings(self, sequences, sizes=None):
        """
        Given a list of numeric sequences, returns the corresponding strings.

        Args:
            sequences (list): A list of sequences of integers.
            sizes (optional): A list of the size of each sequence in the mini-batch.

        Returns:
            A list of sequences of the model's best guess for the transcription.
        """
        strings = []
        for i, sequence in enumerate(sequences):
            size = sizes[i] if sizes else len(sequence)
            string = ''
            for j in range(size):
                char = self.int_to_char[sequence[j].item()]
                if char != self.int_to_char[self.blank_index]:
                    if char == self.labels[self.space_index]:
                        string += ' '
                    else:
                        string += char
            strings.append([string])  # Only one path is returned
        return strings

    def decode(self, probs, sizes=None):
        """
        Returns the argmax decoding given the probability matrix.

        Args:
            probs (tensor): A tensor of character probabilities where probs[c, t]
                is the probability of character c at time t.
            sizes (optional): A list of the size of each sequence in the mini-batch.

        Returns:
            A list of sequences of the model's best guess for the transcription.
        """
        _, max_probs = torch.max(probs, 2)
        strings = self.convert_to_strings(max_probs, sizes)
        return strings