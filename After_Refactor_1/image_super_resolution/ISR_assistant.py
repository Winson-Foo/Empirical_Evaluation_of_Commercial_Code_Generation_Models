import os
from importlib import import_module
import numpy as np

from ISR.utils.utils import setup, parse_args
from ISR.utils.logger import get_logger
from ISR.models import create_model
from ISR.predict.predictor import Predictor
from ISR.train.trainer import Trainer

class Config:
    def __init__(self, config_file, default=False, training=False, prediction=False):
        self.config_file = config_file
        self.default = default
        self.training = training
        self.prediction = prediction
        self.session_type, self.generator, self.conf, self.dataset = setup(
            self.config_file, self.default, self.training, self.prediction
        )
        self.lr_patch_size = self.conf["session"][self.session_type]["patch_size"]
        self.scale = self.conf["generators"][self.generator]["x"]
        self.model_path = self.conf["weights_paths"]["generator"]
        self.hr_patch_size = self.lr_patch_size * self.scale
        self.f_ext = self._get_feature_extractor()
        self.discriminator = self._get_discriminator()
        self.trainer = self._get_trainer()

    def _get_module(self, generator):
        return import_module('ISR.models.' + generator)

    def _get_trainer(self):
        if self.session_type != "training":
            return None

        return Trainer(
            generator=self.model,
            discriminator=self.discriminator,
            feature_extractor=self.f_ext,
            lr_train_dir=self.conf["training_sets"][self.dataset]["lr_train_dir"],
            hr_train_dir=self.conf["training_sets"][self.dataset]["hr_train_dir"],
            lr_valid_dir=self.conf["training_sets"][self.dataset]["lr_valid_dir"],
            hr_valid_dir=self.conf["training_sets"][self.dataset]["hr_valid_dir"],
            learning_rate=self.conf["session"][self.session_type]["learning_rate"],
            loss_weights=self.conf["loss_weights"],
            losses=self.conf["losses"],
            dataname=self.conf["training_sets"][self.dataset]["data_name"],
            log_dirs=self.conf["log_dirs"],
            weights_generator=self.conf["weights_paths"]["generator"],
            weights_discriminator=self.conf["weights_paths"]["discriminator"],
            n_validation=self.conf["session"][self.session_type]["n_validation_samples"],
            flatness=self.conf["session"][self.session_type]["flatness"],
            fallback_save_every_n_epochs=self.conf["session"][self.session_type][
                "fallback_save_every_n_epochs"
            ],
            adam_optimizer=self.conf["session"][self.session_type]["adam_optimizer"],
            metrics=self.conf["session"][self.session_type]["metrics"],
        )

    def _get_discriminator(self):
        if not self.conf["default"]["discriminator"]:
            return None

        return create_model(
            model_type="discriminator",
            patch_size=self.hr_patch_size,
            kernel_size=3,
        )

    def _get_feature_extractor(self):
        if not self.conf["default"]["feature_extractor"]:
            return None
        return create_model(
            model_type="feature_extractor",
            patch_size=self.hr_patch_size,
            layers_to_extract=self.conf["feature_extractor"]["vgg19"][
                "layers_to_extract"
            ],
        )

    def train(self):
        if self.session_type != "training":
            return None
        self.trainer.train(
            epochs=self.conf[self.session_type]["epochs"],
            steps_per_epoch=self.conf[self.session_type]["steps_per_epoch"],
            batch_size=self.conf[self.session_type]["batch_size"],
            monitored_metrics=self.conf[self.session_type]["monitored_metrics"],
        )

    def predict(self):
        if self.session_type != "prediction":
            return None
        pr_h = Predictor(input_dir=self.conf["test_sets"][self.dataset])
        pr_h.get_predictions(gen, self.model_path)

    @property
    def model(self):
        if not hasattr(self, "_model"):
            module = self._get_module(self.generator)
            self._model = module.make_model(
                self.conf["generators"][self.generator], self.lr_patch_size
            )
        return self._model


def run(config_file, default=False, training=False, prediction=False):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logger = get_logger(__name__)
    config = Config(config_file, default, training, prediction)
    config.train()
    config.predict()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(1000)
    run(
        config_file=args["config_file"],
        default=args["default"],
        training=args["training"],
        prediction=args["prediction"],
    )