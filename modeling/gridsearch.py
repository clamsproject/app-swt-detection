import itertools

import modeling.backbones

# parameter values from the best performing models in v5.0
num_splits = {1}
num_epochs = {10}
num_layers = {2}
pos_length = {6000000}
pos_unit = {60000}
dropouts = {0.1}
# img_enc_name = modeling.backbones.model_map.keys()
img_enc_name = {'convnext_lg', 'convnext_tiny'}

pos_abs_th_front = {5}
pos_abs_th_end = {10}
pos_vec_coeff = {0, 0.5}  # when 0, positional encoding is not enabled
block_guids_train = [
    ["cpb-aacip-254-75r7szdz"],     # always block this the most "uninteresting" video (88/882 frames annotated)
]
block_guids_valid = [
    [                               # block all loosely-annotated videos
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
    ]
    # {"cpb-aacip-254-75r7szdz"},  # effectively no block except
]
# we no longer use bins, keeping this just for historical reference
# bins = [{'pre': {'slate': ['S'], 'chyron': ['I', 'N', 'Y'], 'credit': ['C']}}]

param_keys = ['num_splits', 'num_epochs', 'num_layers', 'pos_length', 'pos_unit', 'dropouts', 'img_enc_name', 'pos_abs_th_front', 'pos_abs_th_end', 'pos_vec_coeff', 'block_guids_train', 'block_guids_valid']
l = locals()
configs = []
for vals in itertools.product(*[l[key] for key in param_keys]):
    configs.append(dict(zip(param_keys, vals)))

