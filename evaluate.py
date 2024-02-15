"""Evaluating TimePoints in TimeFrames

Looks at TimePoints within TimeFrames derived by the stitcher, and compares labels
before and after stitching. Note that this ignores TimePoints that may have been
correctly recognized by the classifier but that did not end up being part of a
TimeFrame, so this gives too rosy a picture of the stitcher, would need to run
the app again but then write all TimePoints to the output. It is fair to say that
this evaluation gives a decent representation of the precision.

We have the results of SWT plus annotations files:

- MMIF files with annotations for all time frames with within the timeframes
  time points every 1000 milliseconds
- GBH annotations with approcimately a timepoint for every 2 seconds, but not
  quite, so the annotations and the data do not line up perfectly

Because of the mismatch of system predictions and gold predictions we need to do
one of two things:

- Find an approximate gold timepoint for each timepoint predicted by the system.
- Run the code on the exact timepoints, this would need to be done by making the
  classifier run on specified time points rather than using some sample rate, or
  allow a sampe rate specified in frames.

Since the timepoints never seem to be more out of sync than 32ms, we will go with
the first.

"""

# TODO: this should probably live in the modeling package


import os
import csv
import collections
from mmif import Mmif


# MMIF files
mmif_files = ('ex-aapb-50.json', 'ex-aapb-69.json', 'ex-aapb-75.json')

# Annotation files
gold_files = (
    'modeling/annotations-gbh/cpb-aacip-507-028pc2tp2z.csv',
    'modeling/annotations-gbh/cpb-aacip-690722078b2.csv',
    'modeling/annotations-gbh/cpb-aacip-75-72b8h82x.csv')

# Mappings from raw labels to binned labels
bin_mappings = {
    'B': 'bars',
    'S': 'slate', 'S:H': 'slate', 'S:C': 'slate', 'S:D': 'slate', 'S:G': 'slate',
    'I': 'chyron', 'N': 'chyron', 'Y': 'chyron',
    'C': 'credits',}

# Output files
gold_points_file = 'out-gold-points.tab'
mmif_points_file = 'out-mmif-points.tab'
combined_points_file = 'out-combined.tab'


def evaluate(mmif_file: str, gold_file: str):
    print(mmif_file)
    gold_points = read_gold_points(gold_file)
    mmif_points = read_mmif_points(mmif_file)
    #print(collections.Counter(gold_points.values()))
    #print(collections.Counter(mmif_points.values()))
    combined = combine(gold_points, mmif_points)
    save_points(gold_file, gold_points, mmif_points, combined)
    return calculate_accuracy(combined)


def read_gold_points(gold_file: str):
    points = {}
    with open(gold_file, mode ='r') as fh:
        fh.readline()
        for fname, _seen, label, sublabel, *rest in csv.reader(fh):
            point = fname.split('_')[2][:-4]
            point = int(point[:-2] + '00')
            label = f'{label}:{sublabel}' if sublabel else label
            label = bin_mappings.get(label, 'other')
            points[point] = label
    return points


def read_mmif_points(mmif_file: str):
    mmif_obj = Mmif(open(mmif_file).read())
    for view in mmif_obj.views:
        annotations = view.annotations
        break
    timepoints_idx = {}
    timeframes = []
    print(f'  annotations:    {len(annotations)}')
    for annotation in annotations:
        if 'TimeFrame' in str(annotation.at_type):
            timeframes.append(annotation)
        elif 'TimePoint' in str(annotation.at_type):
            timepoints_idx[annotation.id] = annotation
    mmif_points = {}
    with open('collected_different.txt', 'w') as fh:
        for tf in timeframes:
            post_label = tf.get_property('frameType')
            #print(tf.id, post_label)
            for tp_id in tf.get_property('targets'):
                tp = timepoints_idx[tp_id]
                t = int(tp.get_property('timePoint'))
                pre_label = tp.get_property('label')
                pre_label = bin_mappings.get(pre_label, 'other')
                if pre_label != post_label:
                    fh.write(f'{t} {pre_label} {post_label}\n')
                #print(t, pre_label, post_label)
                mmif_points[t] = (pre_label, post_label)
    return mmif_points


def save_points(gold_file, gold_points, mmif_points, combined):
    fname = os.path.basename(gold_file)
    fname = os.path.splitext(fname)[0]
    with open(gold_points_file, 'a') as fh:
        for key, val in gold_points.items():
            fh.write(f'{fname}\t{key}\t{val}\n')
    with open(mmif_points_file, 'a') as fh:
        for key, val in mmif_points.items():
            fh.write(f'{fname}\t{key}\t{val}\n')
    with open(combined_points_file, 'a') as fh:
        for point in sorted(combined):
            pre_label, post_label, gold_label = combined[point]
            fh.write(f'{fname}\t{point}\t{pre_label}\t{post_label}\t{gold_label}\n')
            #print(point)
        #for key, val in combined.items():
        #    fh.write(f'{fname}\t{key}\t{val}\n')


def combine(gold_points, mmif_points):
    combined = {k: list(v) for k, v in mmif_points.items() if k in gold_points}
    for p in combined:
        combined[p].append(gold_points.get(p))
    return combined


def calculate_accuracy(combined):
    total = 0
    pre_correct = 0
    pre_wrong = 0
    post_correct = 0
    post_wrong = 0
    for pre, post, gold in combined.values():
        total += 1
        if pre == gold:
            pre_correct += 1
        else:
            pre_wrong += 1
        if post == gold:
            post_correct += 1
        else:
            post_wrong += 1
    print(f'  pre stitching:  {pre_correct / total:.2f}')
    print(f'  post stitching: {post_correct / total:.2f}\n')
    return pre_correct, post_correct, total


def reset_output_files():
    for fname in gold_points_file, mmif_points_file, combined_points_file:
        with open(fname, 'w') as fh:
            pass


if __name__ == '__main__':

    print()
    reset_output_files()
    data = zip(mmif_files, gold_files)
    pre_correct = 0
    post_correct = 0
    total = 0
    for mmif_file, gold_file in data:
        results = evaluate(mmif_file, gold_file)
        pre_correct += results[0]
        post_correct += results[1]
        total += results[2]
    print(f'Pre stitching:  {pre_correct / total:.2f}')
    print(f'Post stitching: {post_correct / total:.2f}\n')
    
