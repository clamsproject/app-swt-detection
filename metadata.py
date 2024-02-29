"""
Metadata for the Scenes-with-text app.

"""
import pathlib

import yaml
from clams.app import ClamsApp
from clams.appmetadata import AppMetadata
from mmif import DocumentTypes, AnnotationTypes
from app import default_model_storage, default_config_fname



def appmetadata() -> AppMetadata:
    """
    Function to set app-metadata values and return it as an ``AppMetadata`` obj.
    Read these documentations before changing the code below
    - https://sdk.clams.ai/appmetadata.html metadata specification. 
    - https://sdk.clams.ai/autodoc/clams.appmetadata.html python API
    
    :return: AppMetadata object holding all necessary information.
    """
    preconf = yaml.safe_load(open(default_config_fname))
    available_models = default_model_storage.glob('*.pt')

    metadata = AppMetadata(
        name="Scenes-with-text Detection",
        description="Detects scenes with text, like slates, chyrons and credits.",
        app_license="Apache 2.0",
        identifier="swt-detection",
        url="https://github.com/clamsproject/app-swt-detection"
    )

    labels = ['bars', 'slate', 'chyron', 'credits', 'NEG']
    metadata.add_input(DocumentTypes.VideoDocument, required=True)
    metadata.add_output(AnnotationTypes.TimeFrame, timeUnit='milliseconds', labelset=labels)
    metadata.add_output(AnnotationTypes.TimePoint, timeUnit='milliseconds', labelset=labels)

    # TODO: defaults are the same as in modeling/config/classifier.yml, which is possibly
    # not a great idea, should perhaps read defaults from the configuration file. There is
    # also a movement afoot to get rid of the configuration file.
    metadata.add_parameter(
        name='startAt', type='integer', default=0,
        description='Number of milliseconds into the video to start processing')
    metadata.add_parameter(
        # 10M ms is almost 3 hours, that should do; this is better than sys.maxint
        # (also, I tried using default=None, but that made stopAt a required property)
        name='stopAt', type='integer', default=10000000,
        description='Number of milliseconds into the video to stop processing')
    metadata.add_parameter(
        name='sampleRate', type='integer', default=1000,
        description='Milliseconds between sampled frames')
    metadata.add_parameter(
        name='minFrameScore', type='number', default=0.01,
        description='Minimum score for a still frame to be included in a TimeFrame')
    metadata.add_parameter(
        name='minTimeframeScore', type='number', default=0.50,
        description='Minimum score for a TimeFrame')
    metadata.add_parameter(
        name='minFrameCount', type='integer', default=2,
        description='Minimum number of sampled frames required for a TimeFrame')
    metadata.add_parameter(
        name='modelName', type='string', 
        default=pathlib.Path(preconf['model_file']).stem,
        choices=[m.stem for m in available_models],
        description='model name to use for classification')
    metadata.add_parameter(
        name='useStitcher', type='boolean', default=True,
        description='Use the stitcher after classifying the TimePoints')

    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
