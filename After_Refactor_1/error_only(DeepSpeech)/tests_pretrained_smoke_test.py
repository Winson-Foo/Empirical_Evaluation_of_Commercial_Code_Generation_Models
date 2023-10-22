import os
import unittest
from urllib.error import HTTPError, URLError
import wget
import shutil

from deepspeech_pytorch.configs.inference_config import LMConfig
from deepspeech_pytorch.enums import DecoderType
from tests.smoke_test import DatasetConfig, DeepSpeechSmokeTest

pretrained_urls = [
    'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/an4_pretrained_v3.ckpt',
    'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/librispeech_pretrained_v3.ckpt',
    'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/ted_pretrained_v3.ckpt'
]

lm_path = 'http://www.openslr.org/resources/11/3-gram.pruned.3e-7.arpa.gz'


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
        try:
            wget.download(lm_path)
        except (HTTPError, URLError) as error:
            print(f"Error downloading lm model: {error}")
            return
        for pretrained_url in pretrained_urls:
            try:
                print("Downloading Pre-trained model: ", pretrained_url)
                file_path = wget.download(pretrained_url)
                pretrained_path = os.path.abspath(file_path)
                lm_configs = [
                    LMConfig(),  # Greedy
                    LMConfig(
                        decoder_type=DecoderType.beam
                    ),  # Test Beam Decoder
                    LMConfig(
                        decoder_type=DecoderType.beam,
                        lm_path=os.path.basename(lm_path),
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
                os.remove(pretrained_path)
            except (HTTPError, URLError, IOError, ValueError) as error:
                    print(f"Error: {error}. Failed to download or remove {pretrained_url}")
                    shutil.rmtree(self.target_dir, ignore_errors=True)
                    return


if __name__ == '__main__':
    unittest.main()