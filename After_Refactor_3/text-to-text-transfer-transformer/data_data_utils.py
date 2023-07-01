import gin
import seqio

DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"
DEFAULT_EXTRA_IDS = 100


def get_default_vocabulary():
    return seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH, DEFAULT_EXTRA_IDS)