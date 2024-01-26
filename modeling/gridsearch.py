import itertools

import backbones

num_splits = {30}
num_epochs = {2, 5, 10, 20}
num_layers = {2, 3, 4, 5}
pos_enc_name = {"fractional", "sinusoidal-add", "sinusoidal-concat", "none"}
pos_unit = {1000, 60000}
pos_enc_dim = {128, 256, 512}
dropouts = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}
img_enc_name = backbones.model_map.keys()
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
bins = [
# Defaults
{'pre': {'slate': ['S'], 'chyron': ['I', 'N', 'Y'], 'credit': ['C']}},
{'post': {'slate': ['S'], 'chyron': ['I', 'N', 'Y'], 'credit': ['C']}},
# Other binning strategies
{'pre': {'B': ['B'], 'S': ['S'], 'M': ['M'], 'I': ['I'], 'N': ['N'], 'Y': ['Y'], 'C': ['C']}, 'post': {'slate': ['S'], 'chyron': ['I', 'N', 'Y'], 'credit': ['C']}},
{'pre': {'bars': ['B'], 'slate': ['S', 'S:H', 'S:C', 'S:D', 'S:G'], 'logo': ['L'], 'chyron': ['I', 'N', 'Y'], 'credit': ['C'], 'main_title': ['M'], 'copyright': ['R']}},
{'pre': {'bars': ['B'], 'slate': ['S', 'S:H', 'S:C', 'S:D', 'S:G'], 'chyron': ['I', 'N', 'Y'], 'text-not-chyron': ['K', 'G', 'T', 'F'], 'person-not-chyron': ['P'], 'credits': ['C', 'R']}, 'post': {'slate': ['slate'], 'chyron': ['chyron'], 'credits': ['credits']}},
{'pre': {'bars': ['B'], 'slate': ['S', 'S:H', 'S:C', 'S:D', 'S:B', 'S:G'], 'person-with-text': ['I', 'N', 'E', 'Y', 'K', 'T'], 'person-no-text': ['P'], 'credits': ['C', 'R']}, 'post': {'bars': ['bars'], 'slate': ['slate'], 'person-with-text': ['person-with-text'], 'credits': ['credits']}},
{'pre': {'bars': ['B'], 'slate': ['S', 'S:H', 'S:C', 'S:D', 'S:B', 'S:G'], 'person-with-text': ['I', 'N', 'E', 'K'], 'other-text': ['T', 'F', 'G', 'Y'], 'person-no-text': ['P'], 'credits': ['C', 'R']}, 'post': {'bars': ['bars'], 'slate': ['slate'], 'person-with-text': ['person-with-text'], 'credits': ['credits']}},

{'pre': {'bars': ['B'], 'slate': ['S', 'S:H', 'S:C', 'S:D', 'S:B', 'S:G'], 'person-with-text': ['I', 'N', 'E', 'Y', 'K', 'T'], 'person-no-text': ['P'], 'credits': ['C', 'R']}},
{'post': {'bars': ['B'], 'slate': ['S', 'S:H', 'S:C', 'S:D', 'S:B', 'S:G'], 'person-with-text': ['I', 'N', 'E', 'Y', 'K', 'T'], 'person-no-text': ['P'], 'credits': ['C', 'R']}},
    
{'pre': {'bars': ['B'], 'slate': ['S', 'S:H', 'S:C', 'S:D', 'S:G'], 'other-opening': ['W', 'L', 'O', 'M'], 'chyron': ['I', 'N', 'Y'], 'not-chyron': ['P', 'K', 'G', 'T', 'F'], 'credits': ['C'], 'copyright': ['R']}},
{'post': {'bars': ['B'], 'slate': ['S', 'S:H', 'S:C', 'S:D', 'S:G'], 'other-opening': ['W', 'L', 'O', 'M'], 'chyron': ['I', 'N', 'Y'], 'not-chyron': ['P', 'K', 'G', 'T', 'F'], 'credits': ['C'], 'copyright': ['R']}},
    
{'pre': {'opening': ['B', 'O', 'M', 'L'], 'text': ['S', 'S:H', 'S:C', 'S:D', 'S:B', 'S:G', 'I', 'N', 'E', 'Y', 'K', 'T', 'G', 'F'], 'credits': ['C', 'R'], 'warning': ['W']}},
{'post': {'opening': ['B', 'O', 'M', 'L'], 'text': ['S', 'S:H', 'S:C', 'S:D', 'S:B', 'S:G', 'I', 'N', 'E', 'Y', 'K', 'T', 'G', 'F'], 'credits': ['C', 'R'], 'warning': ['W']}},
    
{'pre': {'bars': ['O'], 'opening_info': ['B', 'M', 'L'], 'person_identification': ['I', 'N', 'Y', 'P'], 'text_frames': ['S', 'S:H', 'S:C', 'S:D', 'S:B', 'S:G', 'K', 'G', 'T', 'F'], 'credits_and_copyright': ['C', 'R'], 'warning': ['W']}},
{'post': {'bars': ['O'], 'opening_info': ['B', 'M', 'L'], 'person_identification': ['I', 'N', 'Y', 'P'], 'text_frames': ['S', 'S:H', 'S:C', 'S:D', 'S:B', 'S:G', 'K', 'G', 'T', 'F'], 'credits_and_copyright': ['C', 'R'], 'warning': ['W']}},
    
{'pre': {'bars': ['B'], 'slate': ['S', 'S:H', 'S:C', 'S:D', 'S:G'], 'warning': ['W'], 'opening': ['O'], 'main_title': ['M'], 'chyron': ['I'], 'credits': ['C'], 'copyright': ['R']}},
{'post': {'bars': ['B'], 'slate': ['S', 'S:H', 'S:C', 'S:D', 'S:G'], 'warning': ['W'], 'opening': ['O'], 'main_title': ['M'], 'chyron': ['I'], 'credits': ['C'], 'copyright': ['R']}},
    
{'pre': {'bars': ['B'], 'slates': ['S', 'S:H', 'S:C', 'S:D', 'S:B', 'S:G'], 'chyron': ['I', 'N', 'Y'], 'warnings': ['W'], 'logos': ['L'], 'openings': ['O', 'M'], 'text_frames': ['E', 'K', 'T', 'G', 'F'], 'person_identification': ['I', 'N', 'Y', 'P'], 'credits_and_copyright': ['C', 'R']}},
{'post': {'bars': ['B'], 'slates': ['S', 'S:H', 'S:C', 'S:D', 'S:B', 'S:G'], 'chyron': ['I', 'N', 'Y'], 'warnings': ['W'], 'logos': ['L'], 'openings': ['O', 'M'], 'text_frames': ['E', 'K', 'T', 'G', 'F'], 'person_identification': ['I', 'N', 'Y', 'P'], 'credits_and_copyright': ['C', 'R']}},
    
{'pre': {'bars': ['B'], 'slate': ['S', 'S:H', 'S:C', 'S:D', 'S:B', 'S:G'], 'person-with-text': ['I', 'N', 'E', 'Y', 'K', 'T'], 'credits': ['C', 'R']}},
{'post': {'bars': ['B'], 'slate': ['S', 'S:H', 'S:C', 'S:D', 'S:B', 'S:G'], 'person-with-text': ['I', 'N', 'E', 'Y', 'K', 'T'], 'credits': ['C', 'R']}},
    
    
{'pre': {'bars': ['B'], 'slate': ['S', 'S:H', 'S:C', 'S:D', 'S:B', 'S:G'], 'person-with-text': ['I', 'N', 'E', 'K'], 'other-text': ['T', 'F', 'G', 'Y'], 'person-no-text': ['P'], 'credits': ['C', 'R']}},
{'post': {'bars': ['B'], 'slate': ['S', 'S:H', 'S:C', 'S:D', 'S:B', 'S:G'], 'person-with-text': ['I', 'N', 'E', 'K'], 'other-text': ['T', 'F', 'G', 'Y'], 'person-no-text': ['P'], 'credits': ['C', 'R']}},
    
{'pre': {'chyron': ['I', 'N', 'Y'], 'person-not-chyron': ['E', 'P', 'K']}},
{'post': {'chyron': ['I', 'N', 'Y'], 'person-not-chyron': ['E', 'P', 'K']}},
# Evaluating individual categories (treating as binary classifier)
{'pre': {'copyright': ['C']}},
{'post': {'copyright': ['C']}},
{'pre': {'bars': ['B']}},
{'post': {'bars': ['B']}},
{'pre': {'chyron': ['I', 'N', 'Y']}},
{'post': {'chyron': ['I', 'N', 'Y']}},
{'pre': {'credits': ['C'], 'copyright': ['R']}},
{'post': {'credits': ['C'], 'copyright': ['R']}},
{'pre': {'credits-and-copyright': ['C', 'R']}},
{'post': {'credits-and-copyright': ['C', 'R']}},
{'pre': {'slate': ['S', 'S:H', 'S:C', 'S:D', 'S:B', 'S:G']}},
{'post': {'slate': ['S', 'S:H', 'S:C', 'S:D', 'S:B', 'S:G']}}
]


param_keys = ['num_splits', 'num_epochs', 'num_layers', 'pos_enc_name', 'pos_unit', 'pos_enc_dim', 'dropouts', 'img_enc_name', 'block_guids_train', 'block_guids_valid', 'bins']
l = locals()
configs = []
for vals in itertools.product(*[l[key] for key in param_keys]):
    configs.append(dict(zip(param_keys, vals)))

