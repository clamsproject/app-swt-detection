"""
Metadata for the Scenes-with-text app.

"""
import sys
from pathlib import Path

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata
from mmif import DocumentTypes, AnnotationTypes

from modeling import FRAME_TYPES
import modeling.bins

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
        description='Use the image classifier model to generate TimePoint annotations.')
    metadata.add_parameter(
        name='tpModelName', type='string',
        default='convnext_lg',
        choices=list(set(m.stem.split('.')[1] for m in available_models)),
        description='Model name to use for classification, only applies when `useClassifier=true`.')
    metadata.add_parameter(
        name='tpUsePosModel', type='boolean', default=True,
        description='Use the model trained with positional features, only applies when `useClassifier=true`.')
    metadata.add_parameter(
        name='tpStartAt', type='integer', default=0,
        description='Number of milliseconds into the video to start processing, only applies when `useClassifier=true`.')
    metadata.add_parameter(
        name='tpStopAt', type='integer', default=sys.maxsize,
        description='Number of milliseconds into the video to stop processing, only applies when `useClassifier=true`.')
    metadata.add_parameter(
        name='tpSampleRate', type='integer', default=1000,
        description='Milliseconds between sampled frames, only applies when `useClassifier=true`.')
    metadata.add_parameter(
        name='useStitcher', type='boolean', default=True,
        description='Use the stitcher after classifying the TimePoints.')
    metadata.add_parameter(
        name='tfMinTPScore', type='number', default=0.5,
        description='Minimum score for a TimePoint to be included in a TimeFrame. '
                    'A lower value will include more TimePoints in the TimeFrame '
                    '(increasing recall in exchange for precision). '
                    'Only applies when `useStitcher=true`.')
    metadata.add_parameter(
        name='tfMinTFScore', type='number', default=0.9,
        description='Minimum score for a TimeFrame. '
                    'A lower value will include more TimeFrames in the output '
                    '(increasing recall in exchange for precision). '
                    'Only applies when `useStitcher=true`')
    metadata.add_parameter(
        name='tfMinTFDuration', type='integer', default=5000,
        description='Minimum duration of a TimeFrame in milliseconds, only applies when `useStitcher=true`.')
    metadata.add_parameter(
        name='tfAllowOverlap', type='boolean', default=False,
        description='Allow overlapping time frames, only applies when `useStitcher=true`')
    metadata.add_parameter(
        name='tfDynamicSceneLabels', type='string', multivalued=True, default=['credit', 'credits'],
        description='Labels that are considered dynamic scenes. For dynamic scenes, TimeFrame annotations contains '
                    'multiple representative points to follow any changes in the scene. '
                    'Only applies when `useStitcher=true`')
    metadata.add_parameter(
        name='tfLabelMap', type='map', default=[],
        description=(
            '(See also `tfLabelMapPreset`, set `tfLabelMapPreset=nopreset` to make sure that a preset does not '
            'override `tfLabelMap` when using this) Mapping of a label in the input TimePoint annotations to a new '
            'label of the stitched TimeFrame annotations. Must be formatted as IN_LABEL:OUT_LABEL (with a colon). To '
            'pass multiple mappings, use this parameter multiple times. When two+ TP labels are mapped to a TF  '
            'label, it essentially works as a "binning" operation. If no mapping is used, all the input labels are '
            'passed-through, meaning no change in both TP & TF labelsets. However, when at least one label is mapped, '
            'all the other "unset" labels are mapped to the negative label (`-`) and if `-` does not exist in the TF '
            'labelset, it is added automatically. '
            'Only applies when `useStitcher=true`.'))
    labelMapPresetsReformat = {schname: str([f'`{lbl}`:`{binname}`' 
                                             for binname, lbls in scheme.items() 
                                             for lbl in lbls]) 
                               for schname, scheme in modeling.bins.binning_schemes.items()}
    labelMapPresetsMarkdown = '\n'.join([f"- `{k}`: {v}" for k, v in labelMapPresetsReformat.items()])
    metadata.add_parameter(
        name='tfLabelMapPreset', type='string', default='relaxed',
        choices=list(modeling.bins.binning_schemes.keys()),
        description=f'(See also `tfLabelMap`) Preset alias of a label mapping. If not `nopreset`, this parameter will '
                    f'override the `tfLabelMap` parameter. Available presets are:\n{labelMapPresetsMarkdown}\n\n '
                    f'Only applies when `useStitcher=true`.')

    return metadata

# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
