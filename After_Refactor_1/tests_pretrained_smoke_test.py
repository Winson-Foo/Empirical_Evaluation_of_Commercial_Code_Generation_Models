import os
import unittest
import wget

from deepspeech_pytorch.configs.inference_config import LMConfig
from deepspeech_pytorch.enums import DecoderType
from tests.smoke_test import DatasetConfig, DeepSpeechSmokeTest


PRETRAINED_URLS = [
    'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/an4_pretrained_v3.ckpt',
    'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/librispeech_pretrained_v3.ckpt',
    'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/ted_pretrained_v3.ckpt'
]
LM_PATH = 'http://www.openslr.org/resources/11/3-gram.pruned.3e-7.arpa.gz'


class PretrainedSmokeTest(DeepSpeechSmokeTest):

    def test_pretrained_eval_inference(self):
        # Disabled GPU due to using TravisCI
        cuda, precision = False, 32
        train_manifest, val_manifest, test_manifest = self.download_data(
            DatasetConfig(
                target_dir=self.target_dir,
                manifest_dir=self.manifest_dir
            ),
            folders=False
        )
        self.download_lm()
        for pretrained_path in self.download_pretrained():
            lm_configs = [
                LMConfig(),  # Greedy
                LMConfig(decoder_type=DecoderType.beam),  # Test Beam Decoder
                LMConfig(
                    decoder_type=DecoderType.beam,
                    lm_path=os.path.basename(LM_PATH),
                    alpha=1,
                    beta=1
                )  # Test Beam Decoder with LM
            ]
            for lm_config in lm_configs:
                self.eval_model(
                    model_path=pretrained_path,
                    test_path=test_manifest,
                    cuda=cuda,
                    lm_config=lm_config,
                    precision=precision
                )
                self.inference(
                    test_path=test_manifest,
                    model_path=pretrained_path,
                    cuda=cuda,
                    lm_config=lm_config,
                    precision=precision
                )

    def download_pretrained(self):
        pretrained_paths = []
        for pretrained_url in PRETRAINED_URLS:
            print("Downloading pre-trained model: ", pretrained_url)
            pretrained_filename = wget.download(pretrained_url)
            pretrained_path = os.path.abspath(pretrained_filename)
            pretrained_paths.append(pretrained_path)
        return pretrained_paths

    def download_lm(self):
        print("Downloading language model: ", LM_PATH)
        wget.download(LM_PATH)


if __name__ == '__main__':
    unittest.main()
