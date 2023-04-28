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
# Modified to support pytorch Tensors

import torch
from typing import List, Optional, Tuple


class Decoder:
    """
    Basic decoder class from which all other decoders inherit. Implements several
    helper functions. Subclasses should implement the decode() method.

    Args:
        labels: Mapping from integers to characters.
        blank_index: Index for the blank '_' character. Defaults to 0.
    """

    def __init__(self, labels: List[str], blank_index: int = 0) -> None:
        self.labels = labels
        self.int_to_char = {i: c for i, c in enumerate(labels)}
        self.blank_index = blank_index
        space_index = len(labels)  # To prevent errors in decode, we add an out of bounds index for the space
        if ' ' in labels:
            space_index = labels.index(' ')
        self.space_index = space_index

    def decode(self, character_probabilities: torch.Tensor,
               sequence_lengths: Optional[List[int]] = None) -> Tuple[List[List[str]], Optional[List[List[int]]]]:
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription

        Args:
            character_probabilities: Tensor of character probabilities, where character_probabilities[c,t]
                                      is the probability of character c at time t
            sequence_lengths: Size of each sequence in the mini-batch

        Returns:
            Tuple of two lists:
            - List of the model's best guess for transcriptions
            - (Optional) List of time steps per character predicted
        """
        raise NotImplementedError


class BeamCTCDecoder(Decoder):
    def __init__(self,
                 labels: List[str],
                 lm_path: Optional[str] = None,
                 alpha: float = 0,
                 beta: float = 0,
                 cutoff_top_n: int = 40,
                 cutoff_prob: float = 1.0,
                 beam_width: int = 100,
                 num_processes: int = 4,
                 blank_index: int = 0) -> None:
        super(BeamCTCDecoder, self).__init__(labels)
        try:
            from ctcdecode import CTCBeamDecoder
        except ImportError:
            raise ImportError("BeamCTCDecoder requires paddledecoder package.")
        labels = list(labels)  # Ensure labels are a list before passing to decoder
        self._decoder = CTCBeamDecoder(labels, lm_path, alpha, beta, cutoff_top_n, cutoff_prob, beam_width,
                                       num_processes, blank_index)

    def convert_to_strings(self, decoder_outputs: List[List[Tuple[torch.Tensor, int]]],
                           sequence_lengths: List[List[int]]) -> List[List[str]]:
        results = []
        for b, batch in enumerate(decoder_outputs):
            utterances = []
            for p, utt in enumerate(batch):
                size = sequence_lengths[b][p]
                if size > 0:
                    transcript = ''.join([self.int_to_char[x.item()] for x in utt[0][:size]])
                else:
                    transcript = ''
                utterances.append(transcript)
            results.append(utterances)
        return results

    def convert_tensor(self, offsets: List[List[Tuple[torch.Tensor, int]]],
                       sizes: List[List[int]]) -> List[List[torch.Tensor]]:
        results = []
        for b, batch in enumerate(offsets):
            utterances = []
            for p, utt in enumerate(batch):
                size = sizes[b][p]
                if sizes[b][p] > 0:
                    utterances.append(utt[0][:size])
                else:
                    utterances.append(torch.tensor([], dtype=torch.int))
            results.append(utterances)
        return results

    def decode(self, character_probabilities: torch.Tensor,
               sequence_lengths: Optional[List[int]] = None) -> Tuple[List[List[str]], Optional[List[List[int]]]]:
        """
        Decodes probability output using ctcdecode package.

        Args:
            character_probabilities: Tensor of character probabilities, where character_probabilities[c,t]
                                      is the probability of character c at time t
            sequence_lengths: Size of each sequence in the mini-batch

        Returns:
            Tuple of two lists:
            - List of the model's best guess for transcriptions
            - (Optional) List of time steps per character predicted
        """
        character_probabilities = character_probabilities.cpu()
        decoder_outputs, _, offsets, seq_lens = self._decoder.decode(character_probabilities, sequence_lengths)

        strings = self.convert_to_strings(decoder_outputs, seq_lens)
        offsets = self.convert_tensor(offsets, seq_lens)
        return strings, offsets


class GreedyDecoder(Decoder):
    def __init__(self, labels: List[str], blank_index: int = 0) -> None:
        super(GreedyDecoder, self).__init__(labels, blank_index)

    def convert_to_strings(self, decoder_outputs: torch.Tensor,
                           sequence_lengths: Optional[List[int]] = None) -> List[List[str]]:
        """Given a tensor of numeric sequences, returns the corresponding strings"""
        strings = []
        for x in range(len(decoder_outputs)):
            seq_len = sequence_lengths[x] if sequence_lengths is not None else len(decoder_outputs[x])
            string = ''.join([self.int_to_char[decoder_outputs[x][i].item()]
                              for i in range(seq_len)
                              if decoder_outputs[x][i].item() != self.blank_index])
            strings.append([string])  # We only return one path
        return strings

    def decode(self, character_probabilities: torch.Tensor,
               sequence_lengths: Optional[List[int]] = None) -> Tuple[List[List[str]], Optional[List[List[int]]]]:
        """
        Returns the argmax decoding given the probability matrix. Removes
        repeated elements in the sequence, as well as blanks.

        Args:
            character_probabilities: Tensor of character probabilities from the network.
                                       Expected shape of batch x seq_length x output_dim
            sequence_lengths: Size of each sequence in the mini-batch

        Returns:
            Tuple of two lists:
            - List of the model's best guess for transcriptions
            - (Optional) List of time steps per character predicted
        """
        _, max_probs = torch.max(character_probabilities, 2)
        strings = self.convert_to_strings(max_probs.view(max_probs.size(0), max_probs.size(1)),
                                          sequence_lengths)
        return strings, None