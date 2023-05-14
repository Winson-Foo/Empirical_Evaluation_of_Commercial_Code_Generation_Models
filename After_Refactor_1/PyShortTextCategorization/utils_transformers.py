import warnings
from typing import List, Union

import numpy as np
import torch
from transformers import BertTokenizer, BertModel


class BERTObject:
    """ The base class for BERT model that contains the embedding model and the tokenizer. """
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        tokenizer_name: str = "bert-base-uncased",
        trainable: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        """ Initialize a BERT model and tokenizer instance.

        :param model_name: The name of the pretrained BERT model to use (default: bert-base-uncased)
        :param tokenizer_name: The name of the pretrained tokenizer to use (default: bert-base-uncased)
        :param trainable: Whether the model should be trainable (default: False)
        :param device: The device to use for computation (default: cpu)
        """
        if not isinstance(device, torch.device):
            device = torch.device(device)

        if device.type == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA is not available. Device set to 'cpu'.")
            device = torch.device("cpu")

        self.device = device
        self.trainable = trainable

        self.model = BertModel.from_pretrained(model_name, output_hidden_states=True).to(device)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)

        self.num_hidden_layers = self.model.config.num_hidden_layers


class WrappedBERTEncoder(BERTObject):
    """ A class for encoding sentences with BERT models. """
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 48,
        nb_encoding_layers: int = 4,
        trainable: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        """ Initialize a BERT encoder instance.

        :param model_name: The name of the pretrained BERT model to use (default: bert-base-uncased)
        :param tokenizer_name: The name of the pretrained tokenizer to use (default: bert-base-uncased)
        :param max_length: The maximum number of tokens of each sentence (default: 48)
        :param nb_encoding_layers: The number of encoding layers to use (default: 4)
        :param trainable: Whether the model should be trainable (default: False)
        :param device: The device to use for computation (default: cpu)
        """
        super().__init__(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            trainable=trainable,
            device=device,
        )
        self.max_length = max_length
        self.nb_encoding_layers = nb_encoding_layers

    def encode_sentences(
        self, sentences: List[str], numpy: bool = False
    ) -> Union[
        List[np.ndarray, np.ndarray, List[str]],
        List[torch.Tensor, torch.Tensor, List[str]],
    ]:
        """ Encode the given list of sentences into numerical vectors.

        :param sentences: The list of strings to encode
        :param numpy: Whether to output a numpy array if True (default: False)
        :return: The encoded vectors for the sentences
        """
        input_ids = []
        tokenized_texts = []

        for sentence in sentences:
            marked_text = f"{self.tokenizer.cls_token} {sentence} {self.tokenizer.sep_token}"
            encoded_dict = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )

            tokenized_texts.append(self.tokenizer.tokenize(marked_text))
            input_ids.append(encoded_dict["input_ids"])

        input_ids = torch.cat(input_ids, dim=0)
        segments_id = torch.LongTensor(np.array(input_ids > 0))
        input_ids = input_ids.to(self.device)
        segments_id = segments_id.to(self.device)

        with torch.set_grad_enabled(self.trainable):
            output = self.model(input_ids, segments_id)

        sentences_embeddings = output[1]
        hidden_state = output[2]

        all_layers_token_embeddings = torch.stack(hidden_state, dim=0)
        all_layers_token_embeddings = all_layers_token_embeddings.permute(
            1, 2, 0, 3
        )  # swap dimensions to [sentence, tokens, hidden layers, features]
        processed_embeddings = all_layers_token_embeddings[
            :, :, (self.num_hidden_layers + 1 - self.nb_encoding_layers) :, :
        ]
        token_embeddings = torch.reshape(
            processed_embeddings, (len(sentences), self.max_length, -1)
        )

        if numpy:
            sentences_embeddings = sentences_embeddings.detach().numpy()
            token_embeddings = token_embeddings.detach().numpy()

        return sentences_embeddings, token_embeddings, tokenized_texts