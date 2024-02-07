# Scenes-with-text 


## Description

Proof of concept prototype for an app that extracts scenes with textual content. The default model included in the app extracts slates, chyrons and credits.


## User instructions

General user instructions for CLAMS apps are available at the [CLAMS Apps documentation](https://apps.clams.ai/clamsapp).


### System requirements

The preferred platform is Debian 10.13 or higher, but the code is known to run on MacOSX. GPU is not required but performance will be better with it. The main system packages needed are FFmpeg ([https://ffmpeg.org/](https://ffmpeg.org/)), OpenCV4 ([https://opencv.org/](https://opencv.org/)), and Python 3.8 or higher. 

The easiest way to get these is to get the Docker [clams-python-opencv4](https://github.com/clamsproject/clams-python/pkgs/container/clams-python-opencv4) base image. For more details take a peek at the following container specifications for the CLAMS [base]((https://github.com/clamsproject/clams-python/blob/main/container/Containerfile)),  [FFMpeg](https://github.com/clamsproject/clams-python/blob/main/container/ffmpeg.containerfile) and [OpenCV](https://github.com/clamsproject/clams-python/blob/main/container/ffmpeg.containerfile) containers. Python packages needed are: clams-python, ffmpeg-python, opencv-python-rolling, torch, torchmetrics, torchvision, av, pyyaml and tqdm. Some of these are installed on the Docker [clams-python-opencv4](https://github.com/clamsproject/clams-python/pkgs/container/clams-python-opencv4) base image and some are listed in `requirements-app.txt` in this repository.


### Configurable runtime parameters

Apps can be configured at request time using [URL query strings](https://en.wikipedia.org/wiki/Query_string). For runtime parameter supported by this app, please visit the [CLAMS App Directory](https://apps.clams.ai) and look for the app name and version. 


### Running the application

To build the Docker image and run the container

```bash
docker build -t app-swt -f Containerfile .
docker run --rm -d -v /Users/Shared/archive/:/data -p 5000:5000 app-swt
```

The path `/Users/Shared/archive/` should be edited to match your local configuaration.

Using the app to process a MMIF file:

```bash
curl -X POST -d@example-mmif.json http://localhost:5000/
```

This may take a while depending on the size of the video file embedded in the MMIF file. It should return a MMIF object with TimeFrame and TimePoint annotations added.


### Output details

A TimeFrame looks as follows (the scores are somewhat condensed for clarity):

```json
{
  "@type": "http://mmif.clams.ai/vocabulary/TimeFrame/v1",
  "properties": {
    "frameType": "bars",
    "score": 0.9999,
    "scores": [0.9998, 0.9999, 0.9998, 0.9999, 0.9999],
    "targets": ["tp_1", "tp_2", "tp_3", "tp_4", "tp_5"],
    "representatives": ["tp_2"],
    "id": "tf_1"
  }
}
```

The `targets` property containes the identifiers of the TimePoints that are included in the TimeFrame, in `scores` we have the TimePoint scores for the "bars" frame type, in `score` we have the average score for the entire TimeFrame, and in `representatives` we have pointers to TimePoints that are considered representative for thie TimeFrame.

Only TimePoints that are included in a TimeFrame will be in the MMIF output, here is one (heavily condensed for clarity and only showing four of the labels):

```json
{
  "@type": "http://mmif.clams.ai/vocabulary/TimePoint/v1",
  "properties": {
    "timePont": 0,
    "label": "B",
    "labels": ["B", "S", "S:H", "S:C"],
    "scores": [0.9998, 5.7532e-08, 2.4712e-13, 1.9209e-12],
    "id": "tp_1"
  }
}
```

The `label` property has the raw label for the TimePoint (which is potentially different from the frameType in the TimeFrame, for one, for the TimeFrame we typically group various raw labels together). In `labels` we have all labels for the TimePoint and in `scores` we have all classifier scores for the labels. 
