"""
CLAMS app to detect scenes with text.

The kinds of scenes that are recognized depend on the model used but typically
include slates, chyrons and credits.

"""

import argparse
import logging
import math
import time
import warnings
from collections import namedtuple
from typing import Union

from clams import ClamsApp, Restifier
from mmif import Mmif, AnnotationTypes, DocumentTypes, Document
from mmif.utils import video_document_helper as vdh
from mmif.utils import sequence_helper as sqh

from metadata import default_model_storage
from modeling.config import bins
from modeling.data_loader import ImageResizeStrategy


class SwtDetection(ClamsApp):

    def __init__(self, log_to_file: bool = False) -> None:
        super().__init__()
        if log_to_file:
            fh = logging.FileHandler(f'{self.__class__.__name__}.log')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(fh)

    def _appmetadata(self):
        # using metadata.py
        pass

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        # parameters here is a "refined" dict, so hopefully its values are properly
        # validated and casted at this point.

        # TODO: fill in `tfLabelMap` parameter value if a preset is used by the user
        # first fill in labelMap parameter value if a preset is used by the user
        label_map = bins.binning_schemes.get(parameters['tfLabelMapPreset'])
        if label_map is None:
            label_map = parameters['tfLabelMap']
        else:
            label_map = {lbl: binname for binname, lbls in label_map.items() for lbl in lbls}
        parameters['tfLabelMap'] = label_map

        for k, v in parameters.items():
            self.logger.debug(f"Final Configuration: {k} :: {v}")
        if parameters.get('useClassifier'):
            self._annotate_timepoints(mmif, **parameters)
        if parameters.get('useStitcher'):
            self._annotate_timeframes(mmif, **parameters)
        return mmif
    
    @staticmethod
    def _get_first_videodocument(mmif: Mmif) -> Union[Document, None]:
        videos = mmif.get_documents_by_type(DocumentTypes.VideoDocument)
        if not videos:
            warnings.warn('There were no video documents referenced in the MMIF file', UserWarning)
            return None
        return videos[0]

    def _annotate_timepoints(self, mmif: Mmif, **parameters) -> Mmif:
        # assuming the app is processing only one video at a time     
        video = self._get_first_videodocument(mmif)
        if video is None:
            return mmif
        
        # isolate this import so that when running in stitcher mode, we don't need to import torch
        from modeling import classify
        import torch

        batch_size = classify.BATCH_SIZE
        vdh.capture(video)
        total_ms = int(vdh.framenum_to_millisecond(video, video.get_property(vdh.FRAMECOUNT_DOCPROP_KEY)))
        start_ms = max(0, parameters['tpStartAt'])
        final_ms = min(total_ms, parameters['tpStopAt'])
        seek_time = 0
        clss_time = 0
        sframe, eframe = [vdh.millisecond_to_framenum(video, p) for p in [start_ms, final_ms]]
        sampled = vdh.sample_frames(sframe, eframe, parameters['tpSampleRate'] / 1000 * video.get_property('fps'))
        self.logger.info(f'Sampled {len(sampled)} frames ' +
                         f'btw {start_ms} - {final_ms} ms (every {parameters["tpSampleRate"]} ms)')
        all_preds = None
        all_positions = []
        t = time.perf_counter()
        # in the following, the .glob() should always return only one, otherwise we have a problem
        ## naming convention from train.py + gridsearch.py = {timestamp}.{backbonename}.{prebinname}.pos{T/F}.pt
        ## right now, `prebinname` is fixed to `nomap` as we don't use prebinning
        model_filestem = next(default_model_storage.glob(
            f"*.{parameters['tpModelName']}.*.pos{'T' if parameters['tpUsePosModel'] else 'F'}.pt")).stem
        self.logger.info(f"Initiating classifier with {model_filestem}")
        classifier = classify.Classifier(default_model_storage / model_filestem,
                                         self.logger.name if self.logger.isEnabledFor(logging.DEBUG) else None)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Classifier initiation took {time.perf_counter() - t:.2f} seconds")
        for i, batch in enumerate(range(0, len(sampled), batch_size)):
            self.logger.info(f"Extracting batch {i + 1} of size {batch_size} from {batch} to {min(batch + batch_size, len(sampled))}")
            batched_sampled = sampled[batch:batch + batch_size]
            positions = [int(vdh.framenum_to_millisecond(video, sample)) for sample in batched_sampled]
            # extract images
            t = time.perf_counter()
            extracted = vdh.extract_frames_as_images(video, batched_sampled, as_PIL=True)
            if not extracted:
                # in a rare case, where the difference between 29.97 and 29.97002997002997 actually matters, we might 
                # get 1+final_frame as the last sample, which will result in an empty list
                continue
            seek_time += time.perf_counter() - t

            # classify images
            t = time.perf_counter()
            # Note that we are not using the built-in "preprocess" of the 
            # CNN models to experiment with different resize strategies.
            # Hence we do this preprocessing manually and when images are 
            # passed to the classifier, they are resized and normalized.
            resizer = ImageResizeStrategy.get_transform_function('distorted')
            resized = torch.stack(tuple(map(resizer, extracted)))
            predictions = classifier.classify_images(resized, positions, total_ms)
            if all_preds is None:
                all_preds = predictions
            else:
                all_preds = torch.cat((all_preds, predictions), dim=0)
            all_positions.extend(positions)
            clss_time += time.perf_counter() - t
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Image extraction took: {seek_time:.2f} seconds\n")
            self.logger.debug(f"Classification took {clss_time:.2f} seconds")

        v = mmif.new_view()
        self.sign_view(v, parameters)
        v.new_contain(
            AnnotationTypes.TimePoint,
            document=video.id, timeUnit='milliseconds', labelset=classifier.training_labels)
        # add classifier results to view
        for position, prediction in zip(all_positions, all_preds):
            timepoint_annotation = v.new_annotation(AnnotationTypes.TimePoint)
            classification = {lbl: prob.item() for lbl, prob in zip(classifier.training_labels, prediction)}
            label = max(classification, key=classification.get)
            timepoint_annotation.add_property('timePoint', position)
            timepoint_annotation.add_property('label', label)
            timepoint_annotation.add_property('classification', classification)

    def _annotate_timeframes(self, mmif: Mmif, **parameters) -> Mmif:
        
        TimeFrameTuple = namedtuple('TimeFrame', 
                                    ['label', 'tf_score', 'targets', 'representatives'])
        tp_view = mmif.get_view_contains(AnnotationTypes.TimePoint)
        if not tp_view:
            self.logger.info("No TimePoint annotations found.")
            return mmif
        tps = list(tp_view.get_annotations(AnnotationTypes.TimePoint))
        did = tps[0].get_property('document')
        self.logger.debug(f"Found {len(tps)} TimePoint annotations.")

        # first, figure out time point sampling rate by looking at the first three annotations
        # why 3? just as a sanity check
        if len(tps) < 3:
            raise ValueError("At least 3 TimePoint annotations are required to stitch.")
        # and then figure out the time point sampling rate
        testsamples = [vdh.convert_timepoint(mmif, tp, 'milliseconds') for tp in tps[:3]]
        if parameters['useClassifier']:
            tp_sampling_rate = parameters['tpSampleRate']
        else:
            tp_sampling_rate = testsamples[1] - testsamples[0]
        tolerance = 1000 / mmif.get_document_by_id(tps[0].get_property('document')).get_property('fps')
        self.logger.debug(f"TimePoint sampling rate 0-1: {tp_sampling_rate}")
        self.logger.debug(f"TimePoint sampling rate 1-2: {testsamples[2] - testsamples[1]}")
        if tp_sampling_rate - (testsamples[2] - testsamples[1]) > tolerance:
            raise ValueError("TimePoint annotations are not uniformly sampled.")

        # next, validate labels in the input annotations
        src_labels = sqh.validate_labelset(tps)

        self.logger.debug(f"Label map: {parameters['tfLabelMap']}")
        label_remapper = sqh.build_label_remapper(src_labels, parameters['tfLabelMap'])

        # then, build the score lists
        label_idx, scores = sqh.build_score_lists([tp.get_property('classification') for tp in tps],
                                                  label_remapper=label_remapper, score_remap_op=max)

        # keep track of the timepoints that have been included as TF targets
        used_timepoints = set()

        def has_overlapping_timeframes(timepoints: list):
            """
            Given a list of TPs, return True if there is a TP in the list that has already been used.
            """
            for timepoint in timepoints:
                if timepoint in used_timepoints:
                    return True
            return False

        all_tf = []
        # and stitch the scores
        for label, lidx in label_idx.items():
            if label == sqh.NEG_LABEL:
                continue
            stitched = sqh.smooth_outlying_short_intervals(
                scores[lidx],
                math.ceil(parameters['tfMinTFDuration'] / tp_sampling_rate),
                # 1,  # does not smooth negative intervals
                math.ceil(1000 / tp_sampling_rate),  # smooth negative window shorter than 1 sec
                parameters['tfMinTPScore']
            )
            self.logger.debug(f"\"{label}\" stitched: {' '.join([str((s, e, e - s)) for s, e in stitched])}")
            for positive_interval in stitched:
                tp_scores = scores[lidx][positive_interval[0]:positive_interval[1]]
                tf_score = tp_scores.mean()
                self.logger.debug(f"\"{label}\" interval {positive_interval} score: {tf_score} / {parameters['tfMinTFScore']}")
                rep_idx = tp_scores.argmax() + positive_interval[0]
                if tf_score >= parameters['tfMinTFScore']:
                    target_list = [a.id for a in tps[positive_interval[0]:positive_interval[1]]]
                    if label not in parameters['tfDynamicSceneLabels']:
                        reps = [tps[rep_idx].id]
                    else:
                        # TODO (krim @ 10/28/24): before this was done by picking every third TP regardless of the 
                        # sampling rate, this new impl is sill very arbitrary and should be improved in the future
                        
                        # we pick every TP from 2 * minTFDuration time window
                        rep_gap = 2 * math.ceil(parameters['tfMinTFDuration'] / tp_sampling_rate)
                        reps = list(map(lambda x: x.id, tps[positive_interval[0]:positive_interval[1]:rep_gap]))
                    all_tf.append(TimeFrameTuple(label=label, tf_score=tf_score, targets=target_list,
                                                 representatives=reps))
        if not parameters['tfAllowOverlap']:
            overlap_filter = []
            for tf in sorted(all_tf, key=lambda x: x.tf_score, reverse=True):
                if has_overlapping_timeframes(tf.targets):
                    continue
                for target_id in tf.targets:
                    used_timepoints.add(target_id)
                overlap_filter.append(tf)
            all_tf = overlap_filter

        # finally add everything to the output view
        v = mmif.new_view()
        self.sign_view(v, parameters)
        v.new_contain(AnnotationTypes.TimeFrame, labelset=list(set(label_remapper.values())), document=did)
        # this will not work because tf_10 < tf_2 by string comparison
        # for tf in sorted(all_tf, key=lambda x: x.targets[0]):
        for tf in sorted(all_tf, key=lambda x: int(x.targets[0].split('_')[-1])):
            v.new_annotation(AnnotationTypes.TimeFrame,
                             label=tf.label,
                             classification={tf.label: tf.tf_score},
                             targets=tf.targets,
                             representatives=tf.representatives)


def get_app():
    """
    This function effectively creates an instance of the app class, without any arguments passed in, meaning, any
    external information such as initial app configuration should be set without using function arguments. The easiest
    way to do this is to set global variables before calling this.
    """
    return SwtDetection(log_to_file=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    parsed_args = parser.parse_args()

    app = get_app()

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
