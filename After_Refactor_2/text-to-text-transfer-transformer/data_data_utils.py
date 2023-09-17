import seqio

DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"  # GCS
DEFAULT_EXTRA_IDS = 100


class Vocabulary:
    def __init__(self, spm_path=DEFAULT_SPM_PATH, extra_ids=DEFAULT_EXTRA_IDS):
        self.spm_path = spm_path
        self.extra_ids = extra_ids

    def get_default_vocabulary(self):
        return seqio.SentencePieceVocabulary(self.spm_path, self.extra_ids)