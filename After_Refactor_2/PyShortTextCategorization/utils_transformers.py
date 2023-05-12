import warnings

import numpy as np
import torch
from transformers import BertTokenizer, BertModel


class BertObject:
    """
    The base class for BERT model that contains the embedding model and the tokenizer.
    """

    def __init__(self, model=None, tokenizer=None, trainable=False, device='cpu'):
        """
        Initialize the BERT object.

        :param model: BERT model (default: None, with model `bert-base-uncase` to be used)
        :param tokenizer: BERT tokenizer (default: None, with model `bert-base-uncase` to be used)
        :param trainable: Whether to set the model to trainable mode
        :param device: device the language model is stored (default: `cpu`)
        """
        if device == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                warnings.warn("CUDA is not available. Device set to 'cpu'.")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.is_trainable = trainable

        if model is None:
            self.model = BertModel.from_pretrained('bert-base-uncased',
                                                   output_hidden_states=True)\
                            .to(self.device)
        else:
            self.model = model.to(self.device)

        if self.is_trainable:
            self.model.train()

        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        else:
            self.tokenizer = tokenizer

        self.num_hidden_layers = self.model.config.num_hidden_layers


class WrappedBertEncoder(BertObject):
    """
    This is the class that encodes sentences with BERT models.
    """

    def __init__(
            self,
            model=None,
            tokenizer=None,
            max_length=48,
            num_encoding_layers=4,
            trainable=False,
            device='cpu'
    ):
        """
        Initialize the WrappedBertEncoder object.

        :param model: BERT model (default: None, with model `bert-base-uncase` to be used)
        :param tokenizer: BERT tokenizer (default: None, with model `bert-base-uncase` to be used)
        :param max_length: maximum number of tokens of each sentence (default: 48)
        :param num_encoding_layers: number of encoding layers (taking the last layers to encode the sentences,
         default: 4)
        :param trainable: Whether to set the model to trainable mode
        :param device: device the language model is stored (default: `cpu`)
        """
        super().__init__(model=model, tokenizer=tokenizer, trainable=trainable, device=device)

        self.max_length = max_length
        self.num_encoding_layers = num_encoding_layers

    def encode_sentences(self, sentences, numpy=False):
        """
        Encode the sentences into numerical vectors.

        :param sentences: list of strings to encode
        :param numpy: output a numpy array if `True`; otherwise, output a torch tensor. (Default: `False`)
        :return: encoded vectors for the sentences
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

        if self.is_trainable:
            output = self.model(input_ids, segments_id)
            sentences_embeddings = output[1]
            hidden_states = output[2]
        else:
            with torch.no_grad():
                output = self.model(input_ids, segments_id)
                sentences_embeddings = output[1]
                hidden_states = output[2]

        token_embeddings = self.process_embeddings(hidden_states)

        if numpy:
            sentences_embeddings = sentences_embeddings.detach().numpy()
            token_embeddings = token_embeddings.detach().numpy()

        return sentences_embeddings, token_embeddings, tokenized_texts

    def process_embeddings(self, hidden_states):
        """
        Process the embeddings.

        :param hidden_states: The hidden states of the BERT model
        :return: The processed embeddings
        """
        all_layers_token_embeddings = torch.stack(hidden_states, dim=0)
        all_layers_token_embeddings = all_layers_token_embeddings.permute(
            1, 2, 0, 3)  # swap dimensions to [sentence, tokens, hidden layers, features]
        processed_embeddings = all_layers_token_embeddings[:, :, (self.num_hidden_layers + 1 - self.num_encoding_layers):, :]
        token_embeddings = torch.reshape(processed_embeddings, (len(sentences), self.max_length, -1))
        return token_embeddings