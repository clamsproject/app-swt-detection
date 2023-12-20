"""Stitching module

Used by app.py in the parent directory and by the modeling.classify is it is used
in standalone mode.

See app.py for hints on how to uses this, the main method is create_timeframes(),
which takes a list of predictions from the classifier and creates TimeFrames.

"""


import yaml
from modeling import train, negative_label


class Stitcher:

    def __init__(self, **config):
        self.config = config
        self.model_config = yaml.safe_load(open(config["model_config_file"]))
        self.sample_rate = config.get("sampleRate")
        self.min_frame_score = config.get("minFrameScore")
        self.min_timeframe_score = config.get("minTimeframeScore")
        self.min_frame_count = config.get("minFrameCount")
        self.labels = train.get_final_label_names(self.model_config)
        self.debug = False

    def __str__(self):
        return (f'<Stitcher min_frame_score={self.min_frame_score} '
                + f'min_timeframe_score={self.min_timeframe_score} '
                + f'min_frame_count={self.min_frame_count}>')

    def create_timeframes(self, predictions: list) -> list:
        timeframes = self.collect_timeframes(predictions)
        if self.debug:
            print_timeframes('Collected frames', timeframes)
        timeframes = self.filter_timeframes(timeframes)
        if self.debug:
            print_timeframes('Filtered frames', timeframes)
        timeframes = self.remove_overlapping_timeframes(timeframes)
        if self.debug:
            print_timeframes('Final frames', timeframes)
        return timeframes

    def collect_timeframes(self, predictions: list) -> list:
        """Find sequences of frames for all labels where the score of each frame
        is at least the mininum value as defined in self.min_frame_score."""
        timeframes = []
        open_frames = { label: TimeFrame(label) for label in self.labels}
        for prediction in predictions:
            for i, label in enumerate(prediction.labels):
                if label == negative_label:
                    continue
                score = prediction.data[i]
                if score < self.min_frame_score:
                    if open_frames[label]:
                        timeframes.append(open_frames[label])
                    open_frames[label] = TimeFrame(label)
                else:
                    open_frames[label].add_point(prediction.timepoint, score)
        for label in self.labels:
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


class TimeFrame:

    def __init__(self, label: str):
        self.label = label
        self.points = []
        self.scores = []
        self.start = None
        self.end = None
        self.score = None

    def __len__(self):
        return len(self.points)

    def __nonzero__(self):
        return len(self) != 0

    def __str__(self):
        if self.is_empty():
            return "<TimePoint empty>"
        else:
            return f"<TimeFrame {self.label} {self.points[0]}:{self.points[-1]} score={self.score:0.4f}>"

    def add_point(self, point, score):
        self.points.append(point)
        self.scores.append(score)

    def finish(self):
        """Once all points have been added to a timeframe, use this method to
        calculate the timeframe score from the points and to set start and end."""
        self.score = sum(self.scores) / len(self)
        self.start = self.points[0]
        self.end = self.points[-1]

    def is_empty(self):
        return len(self) == 0


def print_timeframes(header, timeframes: list):
    print(f'\n{header} ({len(timeframes)})')
    for tf in sorted(timeframes, key=lambda tf: tf.start):
        print(tf)
