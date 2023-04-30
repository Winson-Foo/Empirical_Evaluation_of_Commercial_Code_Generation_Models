import logging
import hydra
import torch

from deepspeech_pytorch.configs.inference_config import EvalConfig
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, AudioDataLoader
from deepspeech_pytorch.utils import load_model, load_decoder
from deepspeech_pytorch.validation import run_evaluation

logging.basicConfig(level=logging.INFO)

DEVICE = torch.device("cuda" if cfg.model.cuda else "cpu")
BATCH_SIZE = cfg.batch_size
NUM_WORKERS = cfg.num_workers

def load_test_data(model, cfg):
    logging.info('Loading test data')
    test_dataset = SpectrogramDataset(audio_conf=model.spect_cfg,
                                       input_path=hydra.utils.to_absolute_path(cfg.test_path),
                                       labels=model.labels,
                                       normalize=True)
    test_loader = AudioDataLoader(test_dataset,
                                  batch_size=BATCH_SIZE,
                                  num_workers=NUM_WORKERS)

    return test_loader

@torch.no_grad()
def evaluate(cfg: EvalConfig):
    logging.info('Evaluating model')
    
    try:
        model = load_model(device=DEVICE, model_path=cfg.model.model_path)
        decoder = load_decoder(labels=model.labels, cfg=cfg.lm)
        target_decoder = GreedyDecoder(labels=model.labels, blank_index=model.labels.index('_'))
        test_loader = load_test_data(model, cfg)

        wer, cer = run_evaluation(test_loader=test_loader,
                                   device=DEVICE,
                                   model=model,
                                   decoder=decoder,
                                   target_decoder=target_decoder,
                                   precision=cfg.model.precision)

        logging.info('Test Summary \tAverage WER {wer:.3f}\tAverage CER {cer:.3f}\t'.format(wer=wer, cer=cer))

    except Exception as e:
        logging.error(f"Error occured while evaluating model: {e}")