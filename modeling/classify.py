import logging
import time
from typing import List

import torch
import yaml
from PIL import Image

from modeling import train, data_loader, FRAME_TYPES


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
        self.featurizer = data_loader.FeatureExtractor(
            img_enc_name=model_config["img_enc_name"],
            pos_length=model_config.get("pos_length", 0),
            pos_unit=model_config.get("pos_unit", 0))
        label_count = len(FRAME_TYPES) + 1
        if 'bins' in model_config:
            label_count = len(model_config['bins'].keys()) + 1
        self.classifier = train.get_net(
            in_dim=self.featurizer.feature_vector_dim(),
            n_labels=label_count,
            num_layers=model_config["num_layers"],
            dropout=model_config["dropouts"])
        self.classifier.load_state_dict(torch.load(model_checkpoint))
        self.debug = False
        self.logger = logging.getLogger(logger_name if logger_name else self.__class__.__name__)

    def classify_images(self, images: List[Image.Image], positions: List[int], final_pos: int) -> list:
        """
        Image classification for a set of extract images (in PIL.Image format). 
        Useful with using ``mmif.utils.video_document_handler.extract_frames_as_images()``
        """
        predictions = []
        featurizing_time = 0
        classifier_time = 0
        for pos, img in zip(positions, images):
            t = time.perf_counter()
            features = self.featurizer.get_full_feature_vectors(img, pos, final_pos)
            if self.logger.isEnabledFor(logging.DEBUG):
                featurizing_time += time.perf_counter() - t
            t = time.perf_counter()
            prediction = self.classifier(features).detach()
            prediction = Prediction(pos, self.training_labels, prediction)
            if self.logger.isEnabledFor(logging.DEBUG):
                classifier_time += time.perf_counter() - t
            predictions.append(prediction)
        self.logger.debug(f'Featurizing time: {featurizing_time:.2f} seconds\n')
        self.logger.debug(f'Classifier time: {classifier_time:.2f} seconds\n')
        return predictions


class Prediction:

    """Class to store a prediction from the model. The prediction is stored as a
    Torch tensor, but also as a list with softmaxed values. Softmaxed scores can
    be retrieve using the label.

    timepoint   -  the location of the frame in the video, in milliseconds
    labels      -  labels for the prediction, taken from the classifier model
    tensor      -  the tensor that results from running the model on the features
    data        -  the tensor simplified into a simple list with softmax scores
    annotation  -  a MMIF annotation associated with the prediction

    Note that instances of this class know nothing about binning. The labels that
    they get handed in are the pre-binning labels and those are the ones that the
    classifier calculates scores for. To get post-binning score (scores where the 
    pre-binning scores are summed) use the score_for_labels() method and let the
    caller figure out what labels are post-binned. The annotation variable is not
    used unless this code is embedded in a CLAMS App.

    """

    def __init__(self, timepoint: int, labels: list,
                 prediction: torch.Tensor, data: list = None):
        self.timepoint = timepoint
        self.labels = labels
        self.tensor = prediction
        if data is None:
            # TODO: it seems like the dimension has to be -1 or 0. Minor changes
            # in the updated scores result, not sure which one is better. 
            self.data = torch.nn.Softmax(dim=0)(self.tensor).detach().numpy().tolist()
        else:
            self.data = data
        self.annotation = None

    def __str__(self):
        # TODO: this needs to be generalized to any set of labels
        label_scores = ' '.join(["%.4f" % d for d in self.data[:-1]])
        neg_score = self.data[-1]
        return f'<Prediction {self.timepoint:6} {label_scores} {neg_score:.4f}>'

    def score_for_label(self, label: str):
        """Return the score for a label."""
        return self.data[self.labels.index(label)]

    def score_for_labels(self, labels: list):
        """Return the  score for a list of labels. This is used when the SWT app
        uses postbinning. The score of a list of labels is defined to be the score
        for the highest scoring label."""
        scores = [self.score_for_label(label) for label in labels]
        return max(scores)

    def as_json(self):
        return [self.timepoint, self.tensor.detach().numpy().tolist(), self.data]
