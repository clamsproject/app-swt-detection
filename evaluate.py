"""Timepoint evaluation

We have the results of SWT plus annotations files:

- MMIF files with annotations for all time frames with within the timeframes
  time points every 1000 milliseconds
- GBH annotations with approcimately a timepoint for every 2 seconds, but not
  quite, so the annotations and the data do not line up perfectly

Because of the mismatch of system predictions and gold predictions we need to do
one of two things:

- Find an approximate gold timepoint for each timepoint predicted by the system.
- Run the code on the exact timepoints, this would need to be done 

Since the timepoints never seem to be more out of sync that 32ms, we will go with
the first.

"""

from mmif import Mmif


# MMIF files
mmif_files = ('ex-aapb-50.json', 'ex-aapb-69.json', 'ex-aapb-75.json')

# Annotation files
gold_files = (
    'modeling/annotations-gbh/cpb-aacip-507-028pc2tp2z.csv',
    'modeling/annotations-gbh/cpb-aacip-690722078b2.csv',
    'modeling/annotations-gbh/cpb-aacip-75-72b8h82x.csv')

bin_mappings = {
    'B': 'bars',
    'S': 'slate', 'S:H': 'slate', 'S:C': 'slate', 'S:D': 'slate', 'S:G': 'slate',
    'I': 'chyron', 'N': 'chyron', 'Y': 'chyron',
    'C': 'credits',}


fh = open('collected_different.txt', 'w')


def evaluate(mmif_file: str, gold_file: str):
    read_mmif_points(mmif_file)


def read_mmif_points(mmif_file: str):
    mmif_obj = Mmif(open(mmif_file).read())
    for view in mmif_obj.views:
        annotations = view.annotations
        break
    timepoints_idx = {}
    timeframes = []
    print(mmif_file, len(annotations))
    for annotation in annotations:
        if 'TimeFrame' in str(annotation.at_type):
            timeframes.append(annotation)
        elif 'TimePoint' in str(annotation.at_type):
            timepoints_idx[annotation.id] = annotation
    for tf in timeframes:
        post_label = tf.get_property('frameType')
        #print(tf.id, post_label)
        for tp_id in tf.get_property('targets'):
            tp = timepoints_idx[tp_id]
            t = tp.get_property('timePont')
            pre_label = tp.get_property('label')
            pre_label = bin_mappings.get(pre_label, 'other')
            if pre_label != post_label:
                fh.write(f'{t} {pre_label} {post_label}\n')
            print(t, pre_label, post_label)



if __name__ == '__main__':

    data = zip(mmif_files, gold_files)
    for mmif_file, gold_file in data:
        evaluate(mmif_file, gold_file)
