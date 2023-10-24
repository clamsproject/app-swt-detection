# Binning config files

This folder is for YAML files specifying the binning strategy used by the k-folds swt model. Files can be passed into the model using the optional -b or --bins flag:

```python3 train.py <indir> <featuremodel> <k_fold> -b config/default.yml```

## Format

The config files should have the following (general) format:

```
bins:

    pre:
        first_pre_bin:
            - "X"
            - "Y"
            - "Z"
        second_pre_bin:
            - "A"
        third_pre_bin:
            - "B"

    post:
        first_post_bin:
            - first_pre_bin
        second_post_bin:
            - second_pre_bin
            - third_pre_bin
```

This example first groups labels "X", "Y", and "Z" into one (arbirarily named) bin, "A" into another, and "B" into a third before the model is trained. Then, at evaluation, it subsequently collapses second_pre_bin and third_pre_bin into one category, and first_pre_bin into another.
 
Note that the pre and post sections are both optional (though at least one should be specified; to train and test on the full label set, don't pass in any config file to the training script). For example, if we want to train the model on the full label set and then group the labels as above, we could use the following config:

```
bins:
    post:
        first_post_bin:
            - "X"
            - "Y"
            - "Z"
        second_post_bin:
            - "A"
            - "B"
```

Or, similarly, the config could contain only a `pre` section to evaluate on the pre-binned label set (see `default.yml`).