"""Stitching module

Used by app.py in the parent directory and by modeling.classify if it is used
in standalone mode.

See app.py for hints on how to use this, the main method is create_timeframes(),
which takes a list of predictions from the classifier and creates TimeFrames.

"""
import itertools
import operator

import yaml

from modeling import train, negative_label, static_frames


class Stitcher:

    def __init__(self, **config):
        self.config = config
        self.sample_rate = config.get("sampleRate")
        self.min_frame_score = config.get("minFrameScore")
        self.min_timeframe_score = config.get("minTimeframeScore")
        self.min_frame_count = config.get("minFrameCount")
        self.stitch_labels = config.get("postbin")
        self.allow_overlap = config.get('allowOverlap')
        self.debug = False

    def __str__(self):
        return (f'<Stitcher min_frame_score={self.min_frame_score} '
                + f'min_timeframe_score={self.min_timeframe_score} '
                + f'min_frame_count={self.min_frame_count}>')

    def create_timeframes(self, predictions: list) -> list:
        if self.debug:
            print('>>> TimePoint labels:', ' '.join(list(itertools.chain(self.stitch_labels.values()))))
            print('>>> TimeFrame labels:', ' '.join(list(self.stitch_labels.keys())))
        timeframes = self.collect_timeframes(predictions)
        if self.debug:
            print_timeframes('Collected frames', timeframes)
        timeframes = self.filter_timeframes(timeframes)
        if self.debug:
            print_timeframes('Filtered frames', timeframes)
        if not self.allow_overlap:
            timeframes = self.remove_overlapping_timeframes(timeframes)
        for tf in timeframes:
            tf.set_representatives()
        timeframes = list(sorted(timeframes, key=(lambda tf: tf.start)))
        if self.debug:
            print_timeframes('Final frames', timeframes)
        return timeframes

    def collect_timeframes(self, predictions: list) -> list:
        """Find sequences of frames for all labels where the score of each frame
        is at least the mininum value as defined in self.min_frame_score."""
        if self.debug:
            print('>>> labels', self.stitch_labels)
        timeframes = []
        open_frames = {label: TimeFrame(label, self) for label in self.stitch_labels}
        for prediction in predictions:
            for label in [label for label in self.stitch_labels if label != negative_label]:
                score = self._score_for_label(label, prediction)
                if score < self.min_frame_score:
                    # the second part checks whether there is something in the timeframe
                    if open_frames[label] and open_frames[label][0]:
                        timeframes.append(open_frames[label])
                    open_frames[label] = TimeFrame(label, self)
                else:
                    open_frames[label].add_prediction(prediction, score)
        for label in self.stitch_labels:
            if open_frames[label]:
                timeframes.append(open_frames[label])
        for tf in timeframes:
            tf.finish()
        return timeframes

    def filter_timeframes(self, timeframes: list) -> list:
        """Filter out all timeframes with an average score below the threshold defined
        in the configuration settings."""
        # TODO: this now also uses the minimum number of samples, but maybe do this
        # filtering later in case we want to use short competing timeframes as a way
        # to determine whether another timeframe is viable
        return [tf for tf in timeframes
                if (tf.score > self.min_timeframe_score
                    and len(tf) >= self.min_frame_count)]

    def remove_overlapping_timeframes(self, timeframes: list) -> list:
        all_frames = list(sorted(timeframes, key=lambda tf: tf.score, reverse=True))
        outlawed_timepoints = set()
        final_frames = []
        for frame in all_frames:
            if self.is_included(frame, outlawed_timepoints):
                continue
            final_frames.append(frame)
            for p in range(frame.start, frame.end + self.sample_rate, self.sample_rate):
                outlawed_timepoints.add(p)
        return final_frames

    def is_included(self, frame, outlawed_timepoints: set) -> bool:
        for i in range(frame.start, frame.end + self.sample_rate, self.sample_rate):
            if i in outlawed_timepoints:
                return True
        return False

    def _score_for_label(self, label: str, prediction):
        """Return the score for the label, this is somewhat more complicated when
        postbinning is used."""
        if self.stitch_labels is None:
            return prediction.score_for_label(label)
        else:
            return prediction.score_for_labels(self.stitch_labels[label])


class TimeFrame:

    def __init__(self, label: str, stitcher: Stitcher):
        self.targets = []
        self.label = label
        self.points = []
        self.scores = []
        self.representatives = []
        self.start = None
        self.end = None
        self.score = None

    def __len__(self):
        return len(self.targets)

    def __nonzero__(self):
        return len(self) != 0

    def __getitem__(self, item: int):
        return self.targets[item]

    def __str__(self):
        if self.is_empty():
            return "<TimePoint empty>"
        else:
            score = -1 if self.score is None else self.score
            span = f"{self.points[0]}:{self.points[-1]}"
            return f"<TimeFrame {self.label} {span} score={score:0.4f}>"

    def pp(self):
        print(self)
        for t in self.targets:
            print('  ', t)
        print(self.scores)

    def add_prediction(self, prediction, score):
        self.targets.append(prediction)
        self.points.append(prediction.timepoint)
        self.scores.append(score)
        #print(f"{prediction} {score:.4f} ==> {self}")

    def finish(self):
        """Once all points have been added to a timeframe, use this method to
        calculate the timeframe score from the points and to set start and end."""
        self.score = sum(self.scores) / len(self)
        self.start = self.points[0]
        self.end = self.points[-1]

    def is_empty(self) -> bool:
        return len(self) == 0

    def representative_predictions(self) -> list:
        answer = []
        for rep in self.representatives:
            for pred in self.targets:
                if pred.timepoint == rep:
                    answer.append(pred)
        return answer

    def set_representatives(self):
        """Calculate the representative still frames for the time frame, using a
        couple of simple heuristics and the frame type."""
        representatives = list(zip(self.points, self.scores))
        timepoint, max_value = max(representatives, key=operator.itemgetter(1))
        if self.label in static_frames:
            # for these just pick the one with the highest score
            self.representatives = [timepoint]
        else:
            # throw out the lower values
            # TODO: this may throw out too many time points
            representatives = [(tp, val) for tp, val in representatives if val >= self.score]
            # pick every third frame, which corresponds roughly to one every five seconds
            # (expect when all below-average values bundled together at one end)
            representatives = representatives[0::3]
            self.representatives = [tp for tp, val in representatives]

    def pp(self):
        print(self)
        print('  ', self.points)
        print('  ', self.scores)
        print('  ', self.representatives)


def print_timeframes(header, timeframes: list):
    print(f'\n{header} ({len(timeframes)})')
    for tf in sorted(timeframes, key=lambda tf: tf.start):
        print(tf)
