from collections import defaultdict
import gensim
from .textpreprocessing import tokenize


def generate_gensim_corpora(classdict, preprocess_and_tokenize=tokenize):
    """
    Generate gensim bag-of-words dictionary and corpus.

    :param classdict: A dictionary with keys being the class labels, and the values being the list of short texts,
                      in the same format output by `shorttext.data.data_retrieval`.
    :param preprocess_and_tokenize: Preprocessor function that takes a short sentence and return a list of tokens.
    :return: A tuple consisting of a gensim dictionary, a corpus, and a list of class labels.
    """
    classlabels = sorted(classdict.keys())
    doc = [preprocess_and_tokenize(' '.join(classdict[classlabel]))
           for classlabel in classlabels]
    dictionary = gensim.corpora.Dictionary(doc)
    corpus = [dictionary.doc2bow(doctokens) for doctokens in doc]
    return dictionary, corpus, classlabels


def save_corpus(dictionary, corpus, prefix):
    """
    Save gensim corpus and dictionary.

    :param dictionary: Dictionary to be saved.
    :param corpus: Corpus to be saved.
    :param prefix: Prefix of the files to save.
    :return: None.
    """
    dictionary.save(f"{prefix}_dictionary.dict")
    gensim.corpora.MmCorpus.serialize(f"{prefix}_corpus.mm", corpus)


def load_corpus(prefix):
    """
    Load gensim corpus and dictionary.

    :param prefix: Prefix of the file to be loaded.
    :return: Corpus and dictionary.
    """
    corpus = gensim.corpora.MmCorpus(f"{prefix}_corpus.mm")
    dictionary = gensim.corpora.Dictionary.load(f"{prefix}_dictionary.dict")
    return corpus, dictionary


def update_corpus_labels(dictionary, corpus, newclassdict,
                         preprocess_and_tokenize=tokenize):
    """
    Update corpus with additional training data.

    :param dictionary: Original dictionary.
    :param corpus: Original corpus.
    :param newclassdict: Additional training data.
    :param preprocess_and_tokenize: Preprocessor function that takes a short sentence and returns a list of tokens.
    :return: A tuple consisting of an updated corpus and the new corpus (for updating the model).
    """
    newdoc = [preprocess_and_tokenize(' '.join(newclassdict[classlabel]))
              for classlabel in sorted(newclassdict.keys())]
    newcorpus = [dictionary.doc2bow(doctokens) for doctokens in newdoc]
    corpus += newcorpus

    return corpus, newcorpus


def tokens_to_fracdict(tokens):
    """
    Return normalized bag-of-words (BOW) vectors.

    :param tokens: List of tokens.
    :return: Normalized vectors of counts of tokens as a `dict`.
    """
    cntdict = defaultdict(lambda: 0)
    for token in tokens:
        cntdict[token] += 1
    totalcnt = sum(cntdict.values())
    return {token: float(cnt)/totalcnt for token, cnt in cntdict.items()}