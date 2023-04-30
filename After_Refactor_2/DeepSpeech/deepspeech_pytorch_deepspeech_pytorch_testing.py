import hydra
import torch

from deepspeech_pytorch.configs.inference_config import EvalConfig
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, AudioDataLoader
from deepspeech_pytorch.utils import load_model, load_decoder
from deepspeech_pytorch.validation import run_evaluation


def load_model_and_decoder(model_path: str, lm_cfg: dict, labels: list, device: torch.device) -> tuple:
    """
    Load the DeepSpeech model and decoder.

    Args:
        model_path (str): Path to the saved model file.
        lm_cfg (dict): Language model configuration.
        labels (list): List of labels.
        device (torch.device): Device to load the model to.

    Returns:
        Tuple of (model, decoder).
    """
    model = load_model(device=device, model_path=model_path)
    decoder = load_decoder(labels=labels, cfg=lm_cfg)

    return model, decoder


def load_test_dataset(test_path: str, spect_cfg: dict, labels: list) -> SpectrogramDataset:
    """
    Load the test dataset.

    Args:
        test_path (str): Path to the test dataset.
        spect_cfg (dict): Spectrogram configuration.
        labels (list): List of labels.

    Returns:
        SpectrogramDataset.
    """
    return SpectrogramDataset(audio_conf=spect_cfg, input_path=test_path, labels=labels, normalize=True)


def run_evaluation(model: torch.nn.Module, test_loader: AudioDataLoader, decoder: GreedyDecoder, target_decoder: GreedyDecoder, device: torch.device, precision: str) -> tuple:
    """
    Run the evaluation on the test dataset.

    Args:
        model (torch.nn.Module): Trained DeepSpeech model.
        test_loader (AudioDataLoader): Test data loader.
        decoder (GreedyDecoder): Decoder for decoding model predictions.
        target_decoder (GreedyDecoder): Decoder for decoding target labels.
        device (torch.device): Device to run the evaluation on.
        precision (str): Model precision.

    Returns:
        Tuple of (WER, CER).
    """
    with torch.no_grad():
        wer, cer = run_evaluation(
            test_loader=test_loader,
            device=device,
            model=model,
            decoder=decoder,
            target_decoder=target_decoder,
            precision=precision
        )

    return wer, cer


def print_test_summary(wer: float, cer: float):
    """
    Print the test summary.

    Args:
        wer (float): Word error rate.
        cer (float): Character error rate.
    """
    print('Test Summary \tAverage WER {wer:.3f}\tAverage CER {cer:.3f}\t'.format(wer=wer, cer=cer))


def evaluate(cfg: EvalConfig):
    """
    Load the DeepSpeech model, decoder, and test dataset, and run the evaluation.

    Args:
        cfg (EvalConfig): Evaluation configuration.
    """
    device = torch.device("cuda" if cfg.model.cuda else "cpu")

    model, decoder = load_model_and_decoder(
        model_path=cfg.model.model_path,
        lm_cfg=cfg.lm,
        labels=model.labels,
        device=device
    )

    target_decoder = GreedyDecoder(
        labels=model.labels,
        blank_index=model.labels.index('_')
    )

    test_dataset = load_test_dataset(
        test_path=hydra.utils.to_absolute_path(cfg.test_path),
        spect_cfg=model.spect_cfg,
        labels=model.labels
    )

    test_loader = AudioDataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers
    )

    wer, cer = run_evaluation(
        model=model,
        test_loader=test_loader,
        decoder=decoder,
        target_decoder=target_decoder,
        device=device,
        precision=cfg.model.precision
    )

    print_test_summary(wer, cer)