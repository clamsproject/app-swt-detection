"""
Metadata for the Scenes-with-text app.

"""
import sys
from pathlib import Path

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata
from mmif import DocumentTypes, AnnotationTypes

from modeling import FRAME_TYPES

default_model_storage = Path(__file__).parent / 'modeling/models'


def appmetadata() -> AppMetadata:
    """
    Function to set app-metadata values and return it as an ``AppMetadata`` obj.
    Read these documentations before changing the code below
    - https://sdk.clams.ai/appmetadata.html metadata specification. 
    - https://sdk.clams.ai/autodoc/clams.appmetadata.html python API
    
    :return: AppMetadata object holding all necessary information.
    """

    available_models = default_model_storage.glob('*.pt')

    # This was the most frequent label mapping from the old configuration file,
    # which had default mappings for each model.
    labelMap = [
        "B:bars",
        "S:slate",
        "I:chyron", "N:chyron", "Y:chyron",
        "C:credits", "R:credits",
        "W:other_opening", "L:other_opening", "O:other_opening", "M:other_opening",
        "E:other_text", "K:other_text", "G:other_text", "T:other_text", "F:other_text"]

    metadata = AppMetadata(
        name="Scenes-with-text Detection",
        description="Detects scenes with text, like slates, chyrons and credits. "
                    "This app can run in three modes, depending on `useClassifier`, `useStitcher` parameters. "
                    "When `useClassifier=True`, it runs in the \"TimePoint mode\" and generates TimePoint annotations. "
                    "When `useStitcher=True`, it runs in the \"TimeFrame mode\" and generates TimeFrame annotations "
                    "based on existing TimePoint annotations -- if no TimePoint is found, it produces an error. "
                    "By default, it runs in the 'both' mode and first generates TimePoint annotations and then "
                    "TimeFrame annotations on them.",
        app_license="Apache 2.0",
        identifier="swt-detection",
        url="https://github.com/clamsproject/app-swt-detection"
    )

    metadata.add_input(DocumentTypes.VideoDocument, required=True)
    metadata.add_output(AnnotationTypes.TimeFrame, timeUnit='milliseconds')
    metadata.add_output(AnnotationTypes.TimePoint, 
                        timeUnit='milliseconds', labelset=FRAME_TYPES)

    metadata.add_parameter(
        name='useClassifier', type='boolean', default=True,
        description='Use the image classifier model to generate TimePoint annotations')
    metadata.add_parameter(
        name='tpModelName', type='string',
        default='convnext_lg',
        choices=list(set(m.stem.split('.')[1] for m in available_models)),
        description='model name to use for classification, only applies when `useClassifier=true`')
    metadata.add_parameter(
        name='tpUsePosModel', type='boolean', default=True,
        description='Use the model trained with positional features, only applies when `useClassifier=true`')
    metadata.add_parameter(
        name='tpStartAt', type='integer', default=0,
        description='Number of milliseconds into the video to start processing, only applies when `useClassifier=true`')
    metadata.add_parameter(
        name='tpStopAt', type='integer', default=sys.maxsize,
        description='Number of milliseconds into the video to stop processing, only applies when `useClassifier=true`')
    metadata.add_parameter(
        name='tpSampleRate', type='integer', default=1000,
        description='Milliseconds between sampled frames, only applies when `useClassifier=true`')
    metadata.add_parameter(
        name='useStitcher', type='boolean', default=True,
        description='Use the stitcher after classifying the TimePoints')
    metadata.add_parameter(
        name='tfMinTPScore', type='number', default=0.01,
        description='Minimum score for a TimePoint to be included in a TimeFrame, only applies when `useStitcher=true`')
    metadata.add_parameter(
        name='tfMinTFScore', type='number', default=0.5,
        description='Minimum score for a TimeFrame, only applies when `useStitcher=true`')
    metadata.add_parameter(
        name='tfMinTFDuration', type='integer', default=2000,
        description='Minimum duration of a TimeFrame in milliseconds, only applies when `useStitcher=true`')
    metadata.add_parameter(
        name='tfAllowOverlap', type='boolean', default=True,
        description='Allow overlapping time frames, only applies when `useStitcher=true`')
    metadata.add_parameter(
        name='tfDynamicSceneLabels', type='string', multivalued=True, default=['credit', 'credits'],
        description='Labels that are considered dynamic scenes. For dynamic scenes, TimeFrame annotations contains '
                    'multiple representative points to follow any changes in the scene. '
                    'Only applies when `useStitcher=true`')
    metadata.add_parameter(
        # TODO: do we want to use the old default labelMap from the configuration here or
        # do we truly want an empty mapping and use the pass-through, as hinted at in the
        # description (which is now not in sync with the code).
        name='tfLabelMap', type='map', default=labelMap,
        description=(
            'Mapping of a label in the input annotations to a new label. Must be formatted as '
            'IN_LABEL:OUT_LABEL (with a colon). To pass multiple mappings, use this parameter '
            'multiple times. By default, all the input labels are passed as is, including any '
            'negative labels (with default value being no remapping at all). However, when '
            'at least one label is remapped, all the other "unset" labels are discarded as '
            'a negative label. Only applies when `useStitcher=true`'))

    return metadata

# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
