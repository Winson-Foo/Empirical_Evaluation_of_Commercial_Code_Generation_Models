from collections import defaultdict
import gensim
from .textpreprocessing import tokenize

class GensimCorpora:
    def __init__(self, classdict, preprocess_and_tokenize=tokenize):
        """ Initializes GensimCorpora class

        Given a text data, a dict with keys being the class labels, and the values
        being the list of short texts, in the same format output by `shorttext.data.data_retrieval`,
        generate a gensim dictionary and corpus.

        :param classdict: text data, a dict with keys being the class labels, and each value is a list of short texts
        :param proprocess_and_tokenize: preprocessor function, that takes a short sentence, and return a list of tokens (Default: `shorttext.utils.tokenize`)
        """

        self.classdict = classdict
        self.preprocess_and_tokenize = preprocess_and_tokenize
        self.classlabels = sorted(classdict.keys())
        self.doc = [preprocess_and_tokenize(' '.join(classdict[classlabel])) for classlabel in self.classlabels]
        self.dictionary = gensim.corpora.Dictionary(self.doc)
        self.corpus = [self.dictionary.doc2bow(doctokens) for doctokens in self.doc]

    def save_corpus(self, prefix):
        """ Save gensim corpus and dictionary.

        :param prefix: prefix of the files to save
        :return: None
        :type prefix: str
        """
        self.dictionary.save(prefix+'_dictionary.dict')
        gensim.corpora.MmCorpus.serialize(prefix+'_corpus.mm', self.corpus)

    @staticmethod
    def load_corpus(prefix):
        """ Load gensim corpus and dictionary.

        :param prefix: prefix of the file to load
        :return: corpus and dictionary
        :type prefix: str
        :rtype: tuple
        """
        corpus = gensim.corpora.MmCorpus(prefix+'_corpus.mm')
        dictionary = gensim.corpora.Dictionary.load(prefix+'_dictionary.dict')
        return corpus, dictionary

    def update_corpus_labels(self, newclassdict):
        """ Update corpus with additional training data.
        
        With the additional training data, the dictionary and corpus are updated.
        
        :param newclassdict: additional training data
        :return: None
        :type newclassdict: dict
        """

        newdoc = [self.preprocess_and_tokenize(' '.join(newclassdict[classlabel])) for classlabel in sorted(newclassdict.keys())]
        newcorpus = [self.dictionary.doc2bow(doctokens) for doctokens in newdoc]
        self.corpus += newcorpus

    @staticmethod
    def tokens_to_fracdict(tokens):
        """ Return normalized bag-of-words (BOW) vectors.

        :param tokens: list of tokens.
        :type tokens: list
        :return: normalized vectors of counts of tokens as a `dict`
        :rtype: dict
        """
        cntdict = defaultdict(lambda : 0)
        for token in tokens:
            cntdict[token] += 1
        totalcnt = sum(cntdict.values())
        return {token: float(cnt)/totalcnt for token, cnt in cntdict.items()}