# file: bert.py
import warnings

import numpy as np
import torch
from transformers import BertTokenizer, BertModel


class BertObject:
    def __init__(self, model: torch.nn.Module = None, tokenizer: BertTokenizer = None,
                 trainable: bool = False, device: str = 'cpu'):
        """
        A base class for BERT model that contains the embedding model and the tokenizer.

        :param model: A BERT model (default: None, with model "bert-base-uncase" to be used).
        :param tokenizer: A BERT tokenizer (default: None, with model "bert-base-uncase" to be used).
        :param trainable: A boolean indicating whether to train the model (default: False).
        :param device: A device the language model is stored (default: "cpu").
        """
        if device == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                warnings.warn("CUDA is not available. Device set to 'cpu'.")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.trainable = trainable

        if model is None:
            self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)\
                            .to(self.device)
        else:
            self.model = model.to(self.device)

        if self.trainable:
            self.model.train()

        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        else:
            self.tokenizer = tokenizer

        self.num_hidden_layers = self.model.config.num_hidden_layers


class WrappedBertEncoder(BertObject):
    def __init__(self, model: torch.nn.Module = None, tokenizer: BertTokenizer = None,
                 max_length: int = 48, num_encoding_layers: int = 4,
                 trainable: bool = False, device: str = 'cpu'):
        """
        This is the class that encodes sentences with BERT models.

        :param model: A BERT model (default: None, with model "bert-base-uncase" to be used).
        :param tokenizer: A BERT tokenizer (default: None, with model "bert-base-uncase" to be used).
        :param max_length: A maximum number of tokens of each sentence (default: 48).
        :param num_encoding_layers: A number of encoding layers (taking the last layers to encode the sentences, default: 4).
        :param trainable: A boolean indicating whether to train the model (default: False).
        :param device: A device the language model is stored (default: "cpu").
        """
        super(WrappedBertEncoder, self).__init__(
            model=model,
            tokenizer=tokenizer,
            trainable=trainable,
            device=device
        )
        self.max_length = max_length
        self.num_encoding_layers = num_encoding_layers

    def encode_sentences(self, sentences: list, numpy: bool = False):
        """
        Encode the sentences into numerical vectors, given by a list of strings.

        It can output either torch tensors or numpy arrays.

        :param sentences: A list of strings to encode.
        :param numpy: Output a numpy array if True; otherwise, output a torch tensor. (Default: False).
        :return: Encoded vectors for the sentences.
        """
        input_ids = []
        tokenized_texts = []

        for sentence in sentences:
            marked_text = '[CLS]' + sentence + '[SEP]'

            encoded_dict = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )

            tokenized_texts.append(self.tokenizer.tokenize(marked_text))
            input_ids.append(encoded_dict['input_ids'])

        input_ids = torch.cat(input_ids, dim=0)
        segments_id = torch.LongTensor(np.array(input_ids > 0))
        input_ids = input_ids.to(self.device)
        segments_id = segments_id.to(self.device)

        with torch.no_grad():
            output = self.model(input_ids, segments_id)
            sentences_embeddings = output[1]
            hidden_state = output[2]

        all_layers_token_embeddings = torch.stack(hidden_state, dim=0)
        all_layers_token_embeddings = all_layers_token_embeddings.permute(1, 2, 0, 3)  # swap dimensions to [sentence, tokens, hidden layers, features]
        processed_embeddings = all_layers_token_embeddings[:, :, (self.num_hidden_layers+1-self.num_encoding_layers):, :]

        token_embeddings = torch.reshape(processed_embeddings, (len(sentences), self.max_length, -1))

        if numpy:
            sentences_embeddings = sentences_embeddings.detach().numpy()
            token_embeddings = token_embeddings.detach().numpy()

        return sentences_embeddings, token_embeddings, tokenized_texts