"""
Metadata for the Scenes-with-text app.

"""
import sys
from pathlib import Path

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata
from mmif import DocumentTypes, AnnotationTypes

import modeling.config.bins

default_model_storage = Path(__file__).parent / 'modeling/models'
# read yml files and find prebin options in the default_model_storage
label_set_dict = {}
for yml_file in default_model_storage.glob('*.yml'):
    # because I don't want to install pyyaml just for this
    # and the CICD system only installs standard CLAMS sdk to get the 
    # metadata at the bulid time. I don't want to change the CICD, 
    # we have to manually parse the file to find top-level "prebin" key and values
    # TODO (krim @ 11/5/25): this is very brittle, consider switching to 
    # json to store model config in the future.
    with open(yml_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
    in_prebin = False
    prebin_labels = []
    for line in lines:
        stripped = line.strip()
        if line.startswith('prebin:'):
            in_prebin = True
            continue
        if in_prebin:
            if line == '':
                break
            if line.startswith(' ') and ':' in line:
                # this is the "from" keys of the prebin, which should be the `labelSet` used in classification
                label = line.split(':', 1)[0].strip()
                prebin_labels.append(label)
            elif not line.startswith(' '):
                # end of prebin section
                break
        
    if prebin_labels:
        hashable_labelset = tuple(sorted(prebin_labels))
        label_set_dict[yml_file.stem] = hashable_labelset
# now check if the found label sets are consistent
all_label_sets = set(label_set_dict.values())
if len(all_label_sets) == 1:
    # all label sets are the same
    consistent_label_set = list(all_label_sets.pop())
else:
    consistent_label_set = None


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
    # currently, only one "prebin" is used, so we can specify the labels here
    # when in the future, multiple models are provided with different prebins, 
    # we can't specify the labels here before a model is selected. 
    if consistent_label_set:
        metadata.add_output(AnnotationTypes.TimePoint, timeUnit='milliseconds', labelset=consistent_label_set)
    else:
        metadata.add_output(AnnotationTypes.TimePoint, timeUnit='milliseconds')

    metadata.add_parameter(
        name='useClassifier', type='boolean', default=True,
        description='Use the image classifier model to generate TimePoint annotations.')
    metadata.add_parameter(
        name='tpModelName', type='string',
        default='convnextv2_tiny',
        choices=list(set(m.stem.split('.')[2] for m in available_models)),
        description='Model name to use for classification, only applies when `useClassifier=true`.')
    metadata.add_parameter(
        name='tpModelBatchSize', type='integer', default=200,
        description='Number of images to process in a batch for classification. Smaller batch sizes will use less '
                    'memory but may be slower. The default value of 200 is set to be the safely maximum size for '
                    '"large" model running on desktop-grade GPU (12GB VRAM). Only applies when `useClassifier=true`.')
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
                               for schname, scheme in modeling.config.bins.binning_schemes.items() 
                               if not (schname.startswith('collapse-') or schname == 'ignore-difficulties')}
    labelMapPresetsMarkdown = '\n'.join([f"- `{k}`: {v}" for k, v in labelMapPresetsReformat.items()])
    metadata.add_parameter(
        name='tfLabelMapPreset', type='string', default='relaxed',
        choices=list(modeling.config.bins.binning_schemes.keys()) + ['nopreset'],
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
