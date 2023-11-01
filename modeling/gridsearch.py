import itertools

import backbones

num_splits = {30}
num_epochs = {2, 5, 10, 20}
num_layers = {2, 3, 4, 5}
positional_encoding = {"fractional", "sinusoidal", "none"}
dropouts = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}
backbone_name = backbones.model_map.keys()
block_guids_train = [
    {"cpb-aacip-254-75r7szdz"},     # always block this the most "uninteresting" video (88/882 frames annotated)
]
block_guids_valid = [
    {                               # block all loosely-annotated videos
        "cpb-aacip-254-75r7szdz",
        "cpb-aacip-259-4j09zf95",
        "cpb-aacip-526-hd7np1xn78",
        "cpb-aacip-75-72b8h82x",
        "cpb-aacip-fe9efa663c6",
        "cpb-aacip-f5847a01db5",
        "cpb-aacip-f2a88c88d9d",
        "cpb-aacip-ec590a6761d",
        "cpb-aacip-c7c64922fcd",
        "cpb-aacip-f3fa7215348",
        "cpb-aacip-f13ae523e20",
        "cpb-aacip-e7a25f07d35",
        "cpb-aacip-ce6d5e4bd7f",
        "cpb-aacip-690722078b2",
        "cpb-aacip-e649135e6ec",
        "cpb-aacip-15-93gxdjk6",
        "cpb-aacip-512-4f1mg7h078",
        "cpb-aacip-512-4m9183583s",
        "cpb-aacip-512-4b2x34nt7g",
        "cpb-aacip-512-3n20c4tr34",
        "cpb-aacip-512-3f4kk9534t",
    },
    {"cpb-aacip-254-75r7szdz"},  # effectively no block except
]
bins = [
    {'pre': {'slate': ['S'], 'chyron': ['I', 'N', 'Y'], 'credit': ['C']}},
    {'post': {'slate': ['S'], 'chyron': ['I', 'N', 'Y'], 'credit': ['C']}},
]

param_keys = ['num_splits', 'num_epochs', 'num_layers', 'positional_encoding', 'dropouts', 'backbone_name', 'block_guids_train', 'block_guids_valid', 'bins']
l = locals()
configs = []
for vals in itertools.product(*[l[key] for key in param_keys]):
    configs.append(dict(zip(param_keys, vals)))
