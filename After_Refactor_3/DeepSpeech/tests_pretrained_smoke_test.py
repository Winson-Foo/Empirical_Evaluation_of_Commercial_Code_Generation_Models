import os
import logging
import unittest
import wget

from deepspeech_pytorch.configs.inference_config import LMConfig
from deepspeech_pytorch.enums import DecoderType
from tests.smoke_test import DatasetConfig, DeepSpeechSmokeTest

logger = logging.getLogger(__name__)
pretrained_urls = [
    'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/an4_pretrained_v3.ckpt',
    'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/librispeech_pretrained_v3.ckpt',
    'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/ted_pretrained_v3.ckpt'
]
lm_path = 'http://www.openslr.org/resources/11/3-gram.pruned.3e-7.arpa.gz'


class PretrainedSmokeTest(DeepSpeechSmokeTest):
    """
    Test suite to run smoke tests on the pre-trained models.
    """
    def test_pretrained_eval_inference(self):
        """
        Test the pre-trained models using evaluation and inference methods.
        """
        cuda, precision = False, 32
        train_manifest, val_manifest, test_manifest = self.download_data(
            DatasetConfig(
                target_dir=self.target_dir,
                manifest_dir=self.manifest_dir
            ),
            folders=False
        )
        wget.download(lm_path)

        for pretrained_url in pretrained_urls:
            logger.info(f"Running Pre-trained Smoke test for: {pretrained_url}")
            wget.download(pretrained_url)
            file_path = os.path.basename(pretrained_url)
            pretrained_path = os.path.join(os.getcwd(), file_path)

            lm_configs = [
                LMConfig(),  # Greedy
                LMConfig(decoder_type=DecoderType.beam),  # Test Beam Decoder
                LMConfig(
                    decoder_type=DecoderType.beam,
                    lm_path=os.path.basename(lm_path),
                    alpha=1,
                    beta=1
                )  # Test Beam Decoder with LM
            ]
            
            for lm_config in lm_configs:
                self.run_eval(pretrained_path, test_manifest, lm_config, cuda, precision)
                self.run_inference(pretrained_path, test_manifest, lm_config, cuda, precision)

    def run_eval(self, model_path, test_manifest, lm_config, cuda, precision):
        """
        Evaluate the specified model on the test dataset.
        """
        self.eval_model(
            model_path=model_path,
            test_path=test_manifest,
            cuda=cuda,
            lm_config=lm_config,
            precision=precision
        )

    def run_inference(self, model_path, test_manifest, lm_config, cuda, precision):
        """
        Perform inference on the specified model using the test dataset.
        """
        self.inference(
            test_path=test_manifest,
            model_path=model_path,
            cuda=cuda,
            lm_config=lm_config,
            precision=precision
        )


if __name__ == '__main__':
    unittest.main()