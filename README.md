# Scene-with-text 


## Description

A CLAMS app that extracts scenes with textual content. At the moment, it performs 1) image extraction given sampliing rate, 2) perform image classification to find relevant images and their timestamps, 3) contruct [`TimeFrame`](http://mmif.clams.ai/vocabulary) annotation objects based on classified images and return MMIF output. 


## User instructions

General user instructions for CLAMS apps is available at [CLAMS Apps documentation](https://apps.clams.ai/clamsapp).


### System requirements

Video and image processing, including extraction, is relying on [opencv4](https://opencv.org/). Image classifier is implemented with pytorch. GPU is not required but recommended for boosting clasifier runtime.

See [`requirements.txt`](requirements.txt) for a list of Python dependencies.

### Configurable runtime parameters

None at the moment. 

