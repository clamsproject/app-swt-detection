# Using Parameters for the CLI

Some notes on the CLI parameter mappping code, mostly because it took me some time to get this clear in my head.

It was a bit tricky, both because (1) we want to be clear about the mapping from the ArgumentParser arguments to the ones we hand in to the app and (2) we need to deal with some peculiarities of how an app deals with the parameters.

After some experimentation it turns out that the simplest way is to ping the ClamsApp.annotate() method from the app and hand in only those options given by the user. This took away the need to deal with issue (2).


## Parameters for the app

This is how it works when you call the app on server using cURL. What matters is what parameters are handed to the annotate() method, which has the following signature:

```python
def annotate(self, mmif: Union[str, dict, Mmif], **runtime_params: List[str]) -> str:
```


### Case 1 - Default invocation


Default invocation using all defaults:

```bash
$ curl -X POST -d@example-mmif-local.json http://localhost:5000/
```

The runtime_params directory is an empty dictionary, but what gets handed to _annotate() after refinement is:

```json
{'#RAW#': {},
 'allowOverlap': True,
 'map': ['B:bars', 'S:slate', 'S-H:slate', 'S-C:slate', 'S-D:slate', 'S-G:slate',
         'W:other_opening', 'L:other_opening', 'O:other_opening', 'M:other_opening',
         'I:chyron', 'N:chyron', 'Y:chyron', 'C:credit', 'R:credit',
         'E:other_text', 'K:other_text', 'G:other_text', 'T:other_text', 'F:other_text'],
 'minFrameCount': 2,
 'minFrameScore': 0.01,
 'minTimeframeScore': 0.5,
 'modelName': '20240409-091401.convnext_lg',
 'pretty': False,
 'sampleRate': 1000,
 'startAt': 0,
 'stopAt': 9223372036854775807,
 'useStitcher': True}
```


### Case 2 - Overriding the modelName and pretty values

Now adding pretty print and a non-default and smaller model:

```bash
$ curl -X POST -d@example-mmif-local.json "http://localhost:5000?pretty=True&modelName=20240212-131937.convnext_tiny"
```

We now have something in runtime_parameters:

```json
{
  'pretty': ['True'],
  'modelName': ['20240212-131937.convnext_tiny']
}
```

And the parameters handed to _annotate() are the same except for the values of 'pretty' and 'modelName' and the non-empty '#RAW#' dictionary, which holds the runtime parameters:

```json
{'#RAW#': {'modelName': ['20240212-131937.convnext_tiny'], 'pretty': ['True']},
 'allowOverlap': True,
 'map': ['B:bars', 'S:slate', 'S-H:slate', 'S-C:slate', 'S-D:slate', 'S-G:slate',
         'W:other_opening', 'L:other_opening', 'O:other_opening', 'M:other_opening',
         'I:chyron', 'N:chyron', 'Y:chyron', 'C:credit', 'R:credit',
         'E:other_text', 'K:other_text', 'G:other_text', 'T:other_text', 'F:other_text'],
 'minFrameCount': 2,
 'minFrameScore': 0.01,
 'minTimeframeScore': 0.5,
 'modelName': '20240212-131937.convnext_tiny',
 'pretty': True,
 'sampleRate': 1000,
 'startAt': 0,
 'stopAt': 9223372036854775807,
 'useStitcher': True}
```


### Case 3 - Overriding the map default

Now we go a step further by adding a label mapping:

```bash
$ curl -X POST -d@example-mmif-local.json "http://localhost:5000?pretty=True&modelName=20240212-131937.convnext_tiny&map=S:slate&map=B:bars"
```

```json
{
  'pretty': ['True'],
  'modelName': ['20240212-131937.convnext_tiny'],
  'map': ['S:slate', 'B:bars']
}
```

Note how the 'map' value is a dictionary now (this is an inconsistency in the clams code):

```json
{'#RAW#': {'map': ['S:slate', 'B:bars'],
           'modelName': ['20240212-131937.convnext_tiny'],
           'pretty': ['True']},
 'allowOverlap': True,
 'map': {'B': 'bars', 'S': 'slate'},
 'minFrameCount': 2,
 'minFrameScore': 0.01,
 'minTimeframeScore': 0.5,
 'modelName': '20240212-131937.convnext_tiny',
 'pretty': True,
 'sampleRate': 1000,
 'startAt': 0,
 'stopAt': 9223372036854775807,
 'useStitcher': True}
```

 
## Parameters for the CLI

The main issues are (1) creating an ArgumentParser from the app metadata and (2) making sure that we take only the user-added parameters and make sure they are turned into lists of strings.

Because we are focusing on the user-defined parameters only we do not have to deal with the map inconsistency (this is done by the app code itself).
