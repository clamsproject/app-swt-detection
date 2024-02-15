"""visualize.py

Create an HTML file that visualizes the output of the frame classifier.

Usage:

$ python visualize.py -m <MMIF_FILE>

    MMIF_FILE - An MMIF file that refers to a local MP4 video vile

The styles for the popups are based mostly the second of the two following links:
- https://stackoverflow.com/questions/27004136/position-popup-image-on-mouseover
- https://stackoverflow.com/questions/32153973/how-to-display-popup-on-mouse-over-in-html-using-js-and-css

Some things to change here:

- Ignore large stretches where non-other values are below 0.01,
  but keep those that show up in a gold standard if we have one.
- Include information from the TimeFrames that were generated.
- Label names are now hard-coded.

"""


import os
import sys
import json
import argparse
from datetime import datetime
from operator import attrgetter

import cv2
from mmif import Mmif, DocumentTypes, Annotation


# Set to False to save time when frames are already created in a previous run
CREATE_FRAMES = False

# Edit this if we use different labels
# TODO: this should really come from a config file
LABELS = ('slate', 'chyron', 'credits')
LABELS = ('bars', 'slate', 'chyron', 'credits', 'copy', 'text', 'person')
LABELS = ('bars', 'slate', 'chyron', 'credits')


# Mappings from binned labels to raw labels, edit if needed
# TODO: this should also really come from a config file
BIN_MAPPINGS = {
    'bars': ('B'),
    'slate': ('S', 'S:H', 'S:C', 'S:D', 'S:G'),
    'chyron': ('I', 'N', 'Y'),
    'credits': ('C')
}


STYLESHEET = '''
<style>
.none { }
.small { color: #fd0; }
.medium { color: #fa0; }
.large { color: #f60; }
.huge { color: #f20; }
.anchor { color: #666 }
td.popup:hover { z-index: 6; }
td.popup span { position: absolute; left: -9999px; z-index: 6; }
/* Need to change this so that the margin is calculated from the number of columns */
td.popup:hover span { margin-left: 550px; left: 2%; z-index:6; }
</style>
'''


def create_frames(video_file: str, positions: list, frames_dir: str):
    vidcap = cv2.VideoCapture(video_file)
    for milliseconds in positions:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, milliseconds)
        success, image = vidcap.read()
        cv2.imwrite(f"{frames_dir}/frame-{milliseconds:06d}.jpg", image)
        print(milliseconds, success)


def load_annotations(mmif_obj: Mmif):
    timeframes = []
    timepoints = {}
    for view in mmif_obj.views:
        for annotation in view.annotations:
            if 'TimeFrame' in str(annotation.at_type):
                timeframes.append(annotation)
            elif 'TimePoint' in str(annotation.at_type):
                timepoints[annotation.id] = TimePointWrapper(annotation)
    print(f'timeframes: {len(timeframes)}')
    print(f'timepoints: {len(timepoints)}')
    timeframe_wrappers = [TimeFrameWrapper(tf, timepoints) for tf in timeframes]
    return sorted(timeframe_wrappers, key=attrgetter('start')), timepoints


def visualize_mmif(mmif: Mmif, timeframes: list, basename: str, htmlfile: str):
    """Visualize the predictions from a list of predictions."""
    for tf in timeframes:
        print(tf)
    with open(htmlfile, 'w') as fh:
        _print_front_matter(fh, basename)
        for tf in timeframes:
            fh.write(f'\n<p><b>{tf.label}</b> {tf.timeframe.get_property("score"):.4f}</p>\n\n')
            fh.write('<table cellpadding=5 cellspacing=0 border=1>\n')
            fh.write('<tr align="center">\n')
            _print_header(fh, LABELS)
            for tp in tf:
                is_representative = True if tp.id in tf.representatives else False
                _print_row(fh, tp.start, LABELS, tp.classification, is_representative)
            fh.write('</table>\n')
        fh.write('<br/>\n' * 25)
        fh.write('</body>\n')
        fh.write('</html>\n')


def get_color_class(score: float):
    if score > 0.75:
        return "huge"
    elif score > 0.50:
        return "large"
    elif score > 0.25:
        return "medium"
    elif score > 0.01:
        return "small"
    else:
        return "none"


def _print_front_matter(fh, title: str):
    fh.write('<html>\n')
    fh.write(STYLESHEET)
    fh.write('<body>\n')
    fh.write(f'<h2>{title}</h2>\n')


def _print_header(fh, labels: list):
    fh.write('<tr align="center">\n')
    for header in ('ms', 'timstamp') + labels + ('rep', 'img'):
        fh.write(f'  <td>{header}</td>\n')
    fh.write('<tr/>\n')


def _print_row(fh, milliseconds: int, labels: list, scores: dict, is_representative: bool):
    timestamp = millisecond_to_isoformat(milliseconds)
    fh.write('<tr>\n')
    fh.write(f'  <td align="right" class="anchor">{milliseconds}</td>\n')
    fh.write(f'  <td align="right" class="anchor">{timestamp}</td>\n')
    # ignoring the score for the negative label
    for lbl in labels:
        p = scores[lbl]
        url = f"frames/frame-{milliseconds:06}.jpg"
        fh.write(f'  <td align="right" class="{get_color_class(p)}">{p:.4f}</td>\n')
    checkmark = '&#10003;' if is_representative else '&nbsp;'
    fh.write(f'  <td align="center" class="anchor">{checkmark}</td>\n')
    onclick = f"window.open('{url}', '_blank')"
    image = f'<img src="{url}" height="24px">'
    image_popup = f'<img src="{url}">'
    fh.write(
        f'  <td class="popup">\n'
        f'    <a href="#" onClick="{onclick}">{image}</a>\n'
        f'    <span>{image_popup}</span>\n'
        f'  </td>\n')
    fh.write('</tr>\n')


class TimeFrameWrapper:

    """Convenience class to wrap a TimeFrame. Gives easy access to start and end
    offset, labels, and classifications for timepoints."""

    def __init__(self, timeframe: Annotation, timepoints: dict):
        self.timeframe = timeframe
        self.timepoints_idx = timepoints
        self.timepoints = []
        self.start = sys.maxsize
        self.end = 0
        self.label = timeframe.get_property('frameType')
        self.labels = None
        self.targets = timeframe.get_property('targets')
        self.representatives = timeframe.get_property('representatives')
        for tp_id in self.targets:
            tp = timepoints.get(tp_id)
            self.timepoints.append(tp)
            if tp.start < self.start:
                self.start = tp.start
            if tp.start > self.end:
                self.end = tp.start
            if self.labels is None:
                self.labels = tuple(tp.raw_labels)
        #self.fix_representatives()

    def __str__(self):
        cls = self.__class__.__name__
        return f'<{cls} {self.start}:{self.end} {self.label}>'

    def __len__(self):
        return len(self.timepoints)

    def __getitem__(self, idx: int):
        return self.timepoints[idx]

    def positions(self):
        """Returns the positions (in milliseconds) of all timepoints."""
        return [tp.start for tp in self.timepoints]

    def fix_representatives(self):
        """This is a hack to deal with a oversight in the configuration, where bars
        were not added as static timeframes. So we fix that here."""
        # TODO: remove this asap
        if self.label == 'bars':
            tps = sorted(self.timepoints, key=(lambda x: x.classification['bars']))
            self.representatives = [tps[-1].id]


class TimePointWrapper:

    """
    id                  -  identifier, same as identifier of the embedded timepoint
    start               -  start (and end) offset of the timepoint
    raw_label           -  the highest scoring raw label
    raw_labels          -  all raw labels
    raw_label_score     -  the score of the highest scoring raw label
    raw_classification  -  the raw results from the classifier
    label               -  the highest scoring label
    labels              -  all labels
    label_score         -  score of the highest scoring label
    classification      -  the classification after binning
    """

    def __init__(self, timepoint: Annotation):
        self.timepoint = timepoint
        self.id = timepoint.id
        self.start = timepoint.get_property('timePoint')
        self.raw_label = timepoint.get_property('label')
        self.raw_labels = timepoint.get_property('labels')
        self.raw_classification = {}
        for raw_label, score in zip(self.raw_labels, timepoint.get_property('scores')):
            self.raw_classification[raw_label] = score
        self.raw_label_score = self.raw_classification[self.raw_label]
        self.label = None
        self.labels = LABELS
        self.label_score = -1
        self.classification = {}
        for label in LABELS:
            score = 0
            for raw_label in BIN_MAPPINGS[label]:
                score += self.raw_classification[raw_label]
            self.classification[label] = score
            if score > self.label_score:
                self.label = label
                self.label_score = score

    def __str__(self):
        cls = self.__class__.__name__
        raw_label = f'{self.raw_label}={self.raw_label_score:.4f}'
        label = f'{self.label}={self.label_score:.4f}'
        return f'<{cls} {self.start}  {raw_label} {label}>'

def millisecond_to_isoformat(millisecond: int) -> str:
    t = datetime.utcfromtimestamp(millisecond / 1000)
    return t.strftime('%H:%M:%S')


def timepoints_in_timeframes(timeframes: list):
    """All timepoint positions that are included in some timeframe."""
    positions = []
    for tf in timeframes:
        positions.extend(tf.positions())
    return list(sorted(set(positions)))


def missed_timepoints(timepoints, positions):
    answer = []
    print(positions)
    positions = set(positions)
    for tp in timepoints.values():
        for label, score in tp.classification.items():
            if score >= 0.5:
                print(tp)
                answer.append(tp)
    return answer



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", metavar='MMIF_FILE', required=True, help="MMIF file")
    args = parser.parse_args()

    mmif_file = args.m
    mmif_obj = Mmif(open(mmif_file).read())
    video_file = mmif_obj.get_documents_locations(DocumentTypes.VideoDocument)[0]
    basename = os.path.splitext(os.path.basename(mmif_file))[0]
    outdir = os.path.join('html', basename)
    outdir_frames = os.path.join(outdir, 'frames')
    index_file = os.path.join(outdir, f'index-{"-".join(LABELS)}.html')
    os.makedirs(outdir_frames, exist_ok=True)
    timeframes, timepoints = load_annotations(mmif_obj)
    positions = timepoints_in_timeframes(timeframes)
    # This is now useless because all the timepoints we have are the ones inside
    # the timeframes
    # other_timepoints = missed_timepoints(timepoints, positions)
    # positions.extend([tp.start for tp in other_timepoints])
    if CREATE_FRAMES:
        create_frames(video_file, sorted(positions), outdir_frames)
    visualize_mmif(mmif_obj, timeframes, basename, index_file)
