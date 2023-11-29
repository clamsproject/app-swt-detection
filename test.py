"""test.py

Running the application code without using a Flask application.

Usage:

$ python test.py example-mmif.json out.json example-config.yml

"""

import sys

from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes

from classify import Classifier


with open(sys.argv[1]) as fh_in, open(sys.argv[2], 'w') as fh_out:
    
    input_mmif = Mmif(fh_in.read())

    vds = input_mmif.get_documents_by_type(DocumentTypes.VideoDocument)
    if not vds:
        exit("no video found")
    vd = vds[0]

    classifier = Classifier(sys.argv[3])

    # Calculate the frame predictions and extract the timeframes
    predictions = classifier.process_video(vd.location)
    timeframes = classifier.extract_timeframes(predictions)

    # Add the timeframes
    new_view: View = input_mmif.new_view()
    new_view.new_contain(AnnotationTypes.TimeFrame, document=vd.id)
    print(timeframes)
    for tf in timeframes:
        start, end, score, label = tf
        timeframe_annotation = new_view.new_annotation(AnnotationTypes.TimeFrame)
        timeframe_annotation.add_property("start", start)
        timeframe_annotation.add_property("end", end)
        timeframe_annotation.add_property("frameType", label),
        timeframe_annotation.add_property("score", score)

    fh_out.write(input_mmif.serialize(pretty=True))
