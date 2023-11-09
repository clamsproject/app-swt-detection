"""
Metadata for the Scenes-with-text app.

"""

from mmif import DocumentTypes, AnnotationTypes

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata


def appmetadata() -> AppMetadata:
    """
    Function to set app-metadata values and return it as an ``AppMetadata`` obj.
    Read these documentations before changing the code below
    - https://sdk.clams.ai/appmetadata.html metadata specification. 
    - https://sdk.clams.ai/autodoc/clams.appmetadata.html python API
    
    :return: AppMetadata object holding all necessary information.
    """
    
    metadata = AppMetadata(
        name="Scenes-with-text Detection",
        description="Detects scenes with text, like slates, chyrons and credits.",
        app_license="Apache 2.0",
        identifier="swt-detection",
        url="https://github.com/clamsproject/app-swt-detection"
    )

    metadata.add_input(DocumentTypes.VideoDocument, required=True)
    metadata.add_output(AnnotationTypes.TimeFrame)
    
    metadata.add_parameter(name='model', description='the model to use, not implemented yet',
                           type='string', default='vgg16')
    
    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
