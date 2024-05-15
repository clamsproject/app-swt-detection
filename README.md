# Scenes-with-text 


## Description

This app extracts scenes with textual content. It has two parts: (1) a classifier that labels timepoints with categories from a basic set of about two dozen types, and (2) a stitcher that pulls timepoints together into time frames of certain types. For the second part the basic categories can be combined into a less fine-grained set.


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

A TimeFrame looks as follows:

```json
{
  "@type": "http://mmif.clams.ai/vocabulary/TimeFrame/v5",
  "properties": {
    "label": "slate",
    "classification": {
      "slate": 0.9958384416320107
    },
    "targets": [ "tp_31",  "tp_32",  "tp_33",  "tp_34", "tp_35", "tp_36",
                 "tp_37", "tp_38", "tp_39", "tp_40", "tp_41" ],
    "representatives": [ "tp_40" ],
    "id": "tf_2"
  }
}
```

The *label* property has the label of the time frame and the *targets* property contains the identifiers of the TimePoints that are included in the TimeFrame. In *classification* we have the score for the "bars" frame type, which is the average score for all TimePoints in the entire TimeFrame, and in *representatives* we have pointers to TimePoints that are considered representative for the TimeFrame.

The output also has all TimePoints from the document, here is one (heavily condensed for clarity and only showing four of the labels in the *classificatio*n dictionary):

```json
{
  "@type": "http://mmif.clams.ai/vocabulary/TimePoint/v4",
  "properties": {
    "timePoint": 30030,
    "label": "S",
    "classification": {
      "B": 1.3528440945265174e-07,
      "S": 0.9999456405639648,
      "R": 4.139282737014582e-06,
      "NEG": 1.9432637543559395e-07
    },
    "id": "tp_31"
  }
}
```

The *label* property has the raw label for the TimePoint (which is potentially different from the frameType in the TimeFrame, for one, for the TimeFrame we typically group various raw labels together).
