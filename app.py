import argparse
import logging
import math
from typing import Union

from clams import ClamsApp, Restifier
from mmif import Mmif, AnnotationTypes
from mmif.utils import sequence_helper as sqh
from mmif.utils import video_document_helper as vdh

import metadata


class SimpleTimepointsStitcher(ClamsApp):

    def __init__(self):
        super().__init__()

    def _appmetadata(self):
        # using ``metadata.py`` 
        pass

    def _annotate(self, mmif: Union[str, dict, Mmif], **parameters) -> Mmif:
        """
        Reference implementation of the sequence stitching algorithm, replicating "stitcher" in 
        https://apps.clams.ai/swt-detection/v4.2/
        """
        self.logger.info(f"Annotating with parameters: {parameters}")

        v = mmif.new_view()
        self.sign_view(v, parameters)
        
        tp_view = mmif.get_view_contains(AnnotationTypes.TimePoint)
        if not tp_view:
            self.logger.info("No TimePoint annotations found.")
            return mmif
        tps = list(tp_view.get_annotations(AnnotationTypes.TimePoint))
        
        # first, figure out time point sampling rate by looking at the first three annotations
        # why 3? just as a sanity check
        def get_timpoint_ms(tp):
            return vdh.convert_timepoint(mmif, tp, 'milliseconds')
        if len(tps) < 3:
            raise ValueError("At least 3 TimePoint annotations are required to stitch.")
        # 1 frame = ? milliseconds
        tp_sampling_rate = get_timpoint_ms(tps[1]) - get_timpoint_ms(tps[0])
        tolerance = 1000 / mmif.get_document_by_id(tps[0].get_property('document')).get_property('fps')
        self.logger.debug(f"TimePoint sampling rate 0-1: {tp_sampling_rate}")
        self.logger.debug(f"TimePoint sampling rate 1-2: {get_timpoint_ms(tps[2]) - get_timpoint_ms(tps[1])}")
        if tp_sampling_rate - (get_timpoint_ms(tps[2]) - get_timpoint_ms(tps[1])) > tolerance:
            raise ValueError("TimePoint annotations are not uniformly sampled.")

        # next, validate labels in the input annotations
        src_labels = sqh.validate_labelset(tps)
        
        # and build the label remapper
        label_map = metadata.labelMapPresets.get(parameters['labelMapPreset'])
        if label_map is None:
            label_map = parameters['labelMap']
        else:
            label_map = dict([lm.split(':') for lm in label_map])
        self.logger.debug(f"Label map: {label_map}")
        label_remapper = sqh.build_label_remapper(src_labels, label_map)
        
        # then, build the score lists
        label_idx, scores = sqh.build_score_lists([tp.get_property('classification') for tp in tps], 
                                                  label_remapper=label_remapper, score_remap_op=max)
        
        # and stitch the scores
        for label, lidx in label_idx.items():
            if label == sqh.NEG_LABEL:
                continue
            stitched = sqh.smooth_outlying_short_intervals(
                scores[lidx], 
                # parameters['minTFDuration']/1000, 
                math.ceil(parameters['minTFDuration']/tp_sampling_rate),
                1,  # does not smooth negative intervals
                parameters['minTPScore']
            )
            self.logger.debug(f"\"{label}\" stitched: {stitched}")
            for positive_interval in stitched:
                tp_scores = scores[lidx][positive_interval[0]:positive_interval[1]]
                tf_score = tp_scores.mean()
                rep_idx = tp_scores.argmax() + positive_interval[0]
                if tf_score > parameters['minTFScore']:
                    tf = v.new_annotation(AnnotationTypes.TimeFrame)
                    # tf.add_property('labelset', list(label_remapper.values()))
                    tf.add_property('label', label)
                    tf.add_property('classification', {label: tf_score})
                    tf.add_property('targets', [a.long_id for a in tps[positive_interval[0]:positive_interval[1]]])
                    tf.add_property('representatives', [tps[rep_idx].id])
        return mmif


def get_app() -> ClamsApp:
    return SimpleTimepointsStitcher()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")

    parsed_args = parser.parse_args()

    # create the app instance
    app = get_app()

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
