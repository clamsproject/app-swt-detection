"""
Metadata for the Scenes-with-text app.

"""

import pathlib
import sys
import yaml

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata
from mmif import DocumentTypes, AnnotationTypes
from app import default_model_storage#, default_config_fname
from modeling import FRAME_TYPES


def appmetadata() -> AppMetadata:
    """
    Function to set app-metadata values and return it as an ``AppMetadata`` obj.
    Read these documentations before changing the code below
    - https://sdk.clams.ai/appmetadata.html metadata specification. 
    - https://sdk.clams.ai/autodoc/clams.appmetadata.html python API
    
    :return: AppMetadata object holding all necessary information.
    """

    available_models = default_model_storage.glob('*.pt')

    # This was the most frequent label mapping from the now deprecated configuration file,
    # which had default mappings for each model.
    labelMap = [
        "B:bars",
        "S:slate", "S-H:slate", "S-C:slate", "S-D:slate", "S-G:slate",
        "W:other_opening", "L:other_opening", "O:other_opening", "M:other_opening",
        "I:chyron", "N:chyron", "Y:chyron",
        "C:credit", "R:credit",
        "E:other_text", "K:other_text", "G:other_text", "T:other_text", "F:other_text" ]

    metadata = AppMetadata(
        name="Scenes-with-text Detection",
        description="Detects scenes with text, like slates, chyrons and credits.",
        app_license="Apache 2.0",
        identifier="swt-detection",
        url="https://github.com/clamsproject/app-swt-detection"
    )

    metadata.add_input(DocumentTypes.VideoDocument, required=True)
    metadata.add_output(AnnotationTypes.TimeFrame, timeUnit='milliseconds')
    metadata.add_output(AnnotationTypes.TimePoint, 
                        timeUnit='milliseconds', labelset=FRAME_TYPES)

    metadata.add_parameter(
        name='startAt', type='integer', default=0,
        description='Number of milliseconds into the video to start processing')
    metadata.add_parameter(
        name='stopAt', type='integer', default=sys.maxsize,
        description='Number of milliseconds into the video to stop processing')
    metadata.add_parameter(
        name='sampleRate', type='integer', default=1000,
        description='Milliseconds between sampled frames')
    metadata.add_parameter(
        name='minFrameScore', type='number', default=0.01,
        description='Minimum score for a still frame to be included in a TimeFrame')
    metadata.add_parameter(
        name='minTimeframeScore', type='number', default=0.5,
        description='Minimum score for a TimeFrame')
    metadata.add_parameter(
        name='minFrameCount', type='integer', default=2,
        description='Minimum number of sampled frames required for a TimeFrame')
    metadata.add_parameter(
        name='modelName', type='string', 
        default='20240409-091401.convnext_lg',
        choices=[m.stem for m in available_models],
        description='model name to use for classification')
    metadata.add_parameter(
        name='useStitcher', type='boolean', default=True,
        description='Use the stitcher after classifying the TimePoints')
    metadata.add_parameter(
        # TODO: do we want to use the old default labelMap from the configuration here or
        # do we truly want an empty mapping and use the pass-through, as hinted at in the
        # description (which is now not in sync with the code).
        name='map', type='map', default=labelMap,
        description=(
            'Mapping of a label in the input annotations to a new label. Must be formatted as '
            'IN_LABEL:OUT_LABEL (with a colon). To pass multiple mappings, use this parameter '
            'multiple times. By default, all the input labels are passed as is, including any '
            'negative labels (with default value being no remapping at all). However, when '
            'at least one label is remapped, all the other "unset" labels are discarded as '
            'a negative label.'))

    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
