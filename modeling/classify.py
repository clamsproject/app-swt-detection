import logging
import time
from typing import List

import torch
import yaml
from PIL import Image

from modeling import train, data_loader, FRAME_TYPES
from modeling.train import BATCH_SIZE


class Classifier:

    def __init__(self, model_stem, logger_name=None):
        """
        :param model_stem: the stem of the model file, 
                           e.g. "modelpath/model" for "modelpath/model.pt" and "modelpath/model.yml"
        :param logger_name: the name of the logger to use, defaults to the class name
        """
        model_config_file = f"{model_stem}.yml"
        model_checkpoint = f"{model_stem}.pt"
        model_config = yaml.safe_load(open(model_config_file))
        self.training_labels = train.get_prebinned_labelset(model_config)
        self.featurizer = data_loader.FeatureExtractor(**model_config)
        self.featurizer.img_encoder.model.eval()
        label_count = len(FRAME_TYPES) + 1
        if 'bins' in model_config:
            label_count = len(model_config['bins'].keys()) + 1
        self.classifier = train.get_net(
            in_dim=self.featurizer.feature_vector_dim(),
            n_labels=label_count,
            num_layers=model_config["num_layers"],
            dropout=model_config["dropouts"])
        self.classifier.load_state_dict(torch.load(model_checkpoint, weights_only=True))
        self.classifier.eval()
        self.debug = False
        self.logger = logging.getLogger(logger_name if logger_name else self.__class__.__name__)

    def classify_images(self, images: torch.Tensor, positions: List[int], final_pos: int) -> torch.Tensor:
        """
        Image classification for a set of extract images (in PIL.Image format). 
        Useful with using ``mmif.utils.video_document_handler.extract_frames_as_images()``
        """
        featurizing_time = 0
        t = time.perf_counter()
        feat_mat = self.featurizer.get_full_feature_vectors(images, [[pos, final_pos] for pos in positions])
        if self.logger.isEnabledFor(logging.DEBUG):
            featurizing_time += time.perf_counter() - t
        self.logger.debug(f'Featurizing time: {featurizing_time:.2f} seconds\n')
        self.logger.debug(f'Instances: {feat_mat.shape[0]}, Features: {feat_mat.shape[1]}')
        softmax = torch.nn.Softmax(dim=1)
        t = time.perf_counter()
        predictions = self.classifier(feat_mat).detach()
        self.logger.debug(f'Predictions: {predictions.shape}, first: {predictions[0]}')
        probabilities = softmax(predictions)
        # sanity check
        self.logger.debug(f'Probabilities: {probabilities.shape}, first: {probabilities[0]} (sum to {sum(probabilities[0])})')
        self.logger.debug(f'Classifier time: {time.perf_counter() - t:.2f} seconds\n')
        return probabilities
