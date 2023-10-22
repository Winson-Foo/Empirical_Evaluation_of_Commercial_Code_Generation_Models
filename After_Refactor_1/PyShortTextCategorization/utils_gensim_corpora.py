from collections import defaultdict

import gensim

from .textpreprocessing import tokenize


def generate_gensim_corpora(classdict, preprocess_and_tokenize=tokenize):
    """
    Generate gensim bag-of-words dictionary and corpus.

    Given a text data, a dict with keys being the class labels, and the values
    being the list of short texts, in the same format output by `shorttext.data.data_retrieval`,
    return a gensim dictionary and corpus.

    :param classdict: text data, a dict with keys being the class labels, and each value is a list of short texts
    :param preprocess_and_tokenize: preprocessor function, that takes a short sentence, and return a list of tokens (Default: `shorttext.utils.tokenize`)
    :return: a tuple, consisting of a gensim dictionary, a corpus, and a list of class labels
    """
    class_labels = sorted(classdict.keys())
    documents = [preprocess_and_tokenize(' '.join(classdict[class_label])) for class_label in class_labels]
    dictionary = gensim.corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(document_tokens) for document_tokens in documents]
    return dictionary, corpus, class_labels


def save_corpus(dictionary, corpus, prefix):
    """
    Save gensim corpus and dictionary.

    :param dictionary: dictionary to save
    :param corpus: corpus to save
    :param prefix: prefix of the files to save
    :return: None
    """
    dictionary.save(prefix + '_dictionary.dict')
    gensim.corpora.MmCorpus.serialize(prefix + '_corpus.mm', corpus)


def load_corpus(prefix):
    """
    Load gensim corpus and dictionary.

    :param prefix: prefix of the file to load
    :return: corpus and dictionary
    """
    corpus = gensim.corpora.MmCorpus(prefix + '_corpus.mm')
    dictionary = gensim.corpora.Dictionary.load(prefix + '_dictionary.dict')
    return corpus, dictionary


def update_corpus_labels(dictionary, corpus, new_class_dict, preprocess_and_tokenize=tokenize):
    """
    Update corpus with additional training data.

    With the additional training data, the dictionary and corpus are updated.

    :param dictionary: original dictionary
    :param corpus: original corpus
    :param new_class_dict: additional training data
    :param preprocess_and_tokenize: preprocessor function, that takes a short sentence, and return a list of tokens (Default: `shorttext.utils.tokenize`)
    :return: a tuple, an updated corpus, and the new corpus (for updating model)
    """
    new_documents = [preprocess_and_tokenize(' '.join(new_class_dict[class_label])) for class_label in sorted(new_class_dict.keys())]
    new_corpus = [dictionary.doc2bow(document_tokens) for document_tokens in new_documents]
    corpus += new_corpus

    return corpus, new_corpus


def tokens_to_fracdict(tokens):
    """
    Return normalized bag-of-words (BOW) vectors.

    :param tokens: list of tokens.
    :return: normalized vectors of counts of tokens as a `defaultdict`
    """
    count_dict = defaultdict(lambda: 1)
    for token in tokens:
        count_dict[token] += 1
    total_count = sum(count_dict.values())
    return defaultdict(float, {token: count / total_count for token, count in count_dict.items()})