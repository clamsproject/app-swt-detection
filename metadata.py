from mmif import DocumentTypes, AnnotationTypes

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata, Input


# DO NOT CHANGE the function name 
def appmetadata() -> AppMetadata:
    
    metadata = AppMetadata(
        name="Simple Timepoints Stitcher",
        description="Stitches a sequence of `TimePoint` annotations into a sequence of `TimeFrame` annotations, "
                    "performing simple smoothing of short peaks of positive labels.",
        app_license="Apache 2.0", 
        identifier="simple-timepoints-stitcher",  
        url="https://github.com/clamsproject/app-simple-timepoints-stitcher",  
    )
    
    metadata.add_input_oneof(*[Input(at_type=DocumentTypes.AudioDocument), Input(at_type=DocumentTypes.VideoDocument)])
    metadata.add_input(
        AnnotationTypes.TimePoint, timePoint='*', classification='*').description = \
        ('TimePoint annotations to be stitched. Must be "exhaustive" in that it should cover an entire single time '
         'period in the input document, with a uniform sample rate.')
    metadata.add_output(
        AnnotationTypes.TimeFrame, timeUnit='milliseconds', label='*', representatives='*').description = \
        ('Stitched TimeFrame annotations. Each TimeFrame annotation represents a continuous segment of timepoints '
         'and its `label` property is determined by the `labelMap` parameter (see `parameters` section). The '
         '`representatives` is a singleton list of the TimePoint annotation that has the highest score in the '
         'TimeFrame.')
    
    metadata.add_parameter(
        name='labelMap', type='map', default=[],
        description=('mapping of labels in the input annotations to new labels. Must be formatted as '
                     '\"IN_LABEL:OUT_LABEL\" (with a colon). To pass multiple mappings, use this parameter multiple '
                     'times. By default, all the input labels are passed as is, including any \"negative\" labels '
                     '(with default value being no remapping at all). However, when at least one label is remapped, '
                     'all the other \"unset\" labels are discarded asa negative label(\"-\").'))
    metadata.add_parameter(
        name='minTFDuration', type='integer', default=1000,
        description='minimum duration of a TimeFrame in milliseconds')
    metadata.add_parameter(
        name='minTPScore', type='number', default=0.1,
        description='minimum score of a TimePoint to be considered as positive')
    metadata.add_parameter(
        name='minTFScore', type='number', default=0.5,
        description='minimum average score of TimePoints in a TimeFrame to be considered as positive')
    
    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
