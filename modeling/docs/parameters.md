# Using Parameters for the CLI

This is tricky, both because (1) we want to be clear about the mapping from the ArgumentParser arguments to the ones we hand in to the aannotate() method and (2) we need to deal with some peculiarities of how an app deals with the parameters. Issue (2) disappeared with version 1.2.2 of clams-python.


## Parameters for the app

This is how it works when you call the app using cURL. First let's call the SWT app using all defaults:

```
$ curl -X POST -d@example-mmif-local.json http://localhost:5000
```

After this invocation the parameters dictionary handed in to SwtDetection()._annotate() is as follows:

```json
{"#RAW#": {},
 "allowOverlap": true,
 "map": ["B:bars", "S:slate", "S-H:slate", "S-C:slate", "S-D:slate", "S-G:slate",
         "W:other_opening", "L:other_opening", "O:other_opening", "M:other_opening",
         "I:chyron", "N:chyron", "Y:chyron", "C:credit", "R:credit",
         "E:other_text", "K:other_text", "G:other_text", "T:other_text", "F:other_text"],
 "minFrameCount": 2,
 "minFrameScore": 0.01,
 "minTimeframeScore": 0.5,
 "modelName": "20240409-091401.convnext_lg",
 "pretty": false,
 "sampleRate": 1000,
 "startAt": 0,
 "stopAt": 9223372036854775807,
 "useStitcher": true}
```

All the parameters here are refined, that is, the ClamsApp.annotate() method uses a parameter caster to cast the string values from the GET variables to Python types.

To show what happens when we override defaults we now use another (quicker) model, turn on pretty printing, and provide a label mapping:

```
$ curl -X POST -d@example-mmif-local.json \
    "http://localhost:5000?modelName=20240212-131937.convnext_tiny&map=B:bars&map=S:slate&pretty=True"
```

We now have the following parameters handed in to the _annotate() method:

```python
{"#RAW#": {
   "map": ["B:bars", "S:slate"],
   "modelName": ["20240212-131937.convnext_tiny"],
   "pretty": ["True"]},
 "allowOverlap": True,
 "map": {"B": "bars", "S": "slate"},
 "minFrameCount": 2,
 "minFrameScore": 0.01,
 "minTimeframeScore": 0.5,
 "modelName": "20240212-131937.convnext_tiny",
 "pretty": True,
 "sampleRate": 1000,
 "startAt": 0,
 "stopAt": 9223372036854775807,
 "useStitcher": True}
```

There are two changes here: (1) the values for 'modelName', 'pretty' and 'map' are now updated, and (2) a dictionary with the raw parameters are added, where the values are lists of strings.

When the app generates output it will have two fields in the output view's metadata, namely a 'parameters' field and an 'appConfiguration' field. The 'parameters' field has the raw parameters (but not as lists):

```python
"parameters": {
   "modelName": "20240212-131937.convnext_tiny",
   "map": "['B:bars', 'S:slate']",
   "pretty": "True"
```

And the 'appConfiguration' field has the refined parameters:

```python
"appConfiguration": {
  "startAt": 0,
  "stopAt": 9223372036854775807,
  "sampleRate": 1000,
  "minFrameScore": 0.01,
  "minTimeframeScore": 0.5,
  "minFrameCount": 2,
  "modelName": "20240212-131937.convnext_tiny",
  "useStitcher": True,
  "allowOverlap": True,
  "map": { "B": "bars", "S": "slate" },
  "pretty": True,
  "model_file": "/Users/.../app-swt-detection/modeling/models/20240212-131937.convnext_tiny.pt",
  "model_config_file": "/Users/.../app-swt-detection/modeling/models/20240212-131937.convnext_tiny.yml",
  "postbin": {
     "bars": ["B"],
     "slate": ["S"] }
}
```

In the app configuration, the list value of the map was changed into a dictionary and a 'postbin' parametr was added, which is basically an inverted dictionary created from the 'map' parameter (in addition a model configuration file was added).


## Parameters for the CLI

An ArgumentParser is automatically created from the app's metadata. This is all fairly straightforward accept that boolean paramaters from the app are not translated into boolean ArumentParser arguments but rather into a string-valued parameter where the options are restircted to 'True' and 'False'.