"""test.py

Running the application code without using a Flask application.

Usage:

$ python test.py example-mmif.json out.json

"""

import sys

from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes

import classify


with open(sys.argv[1]) as fh_in, open(sys.argv[2], 'w') as fh_out:
    
    input_mmif = Mmif(fh_in.read())

    vds = input_mmif.get_documents_by_type(DocumentTypes.VideoDocument)
    if not vds:
        exit("no video found")
    vd = vds[0]

    # Get the predictions.
    predictions = classify.process_video(vd.location, step=20000)

    # This is silly, we are saving and loading the predictions to and from a file
    # just because downstream functions like enrich_predictions cannot work with
    # the data structures created by process_video().
    classify.save_predictions(predictions, 'preds.json')
    predictions = classify.load_predictions('preds.json')

    # Get the timeframes
    classify.enrich_predictions(predictions)
    timeframes = classify.collect_timeframes(predictions)
    classify.compress_timeframes(timeframes)
    classify.filter_timeframes(timeframes)
    timeframes = classify.remove_overlapping_timeframes(timeframes)

    # Add the timeframes.
    new_view: View = input_mmif.new_view()
    new_view.new_contain(AnnotationTypes.TimeFrame, document=vd.id)
    for tf in timeframes:
        start, end, score, label = tf
        timeframe_annotation = new_view.new_annotation(AnnotationTypes.TimeFrame)
        timeframe_annotation.add_property("start", start)
        timeframe_annotation.add_property("end", end)
        timeframe_annotation.add_property("frameType", label),
        timeframe_annotation.add_property("score", score)

    fh_out.write(input_mmif.serialize(pretty=True))
