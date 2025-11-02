import itertools

import modeling.backbones
import modeling.config.bins
from modeling.config import batches 


## TP classifier training grid search
# Updated based on v8.0 r1, r2, r3 results. 
# Best overall config: convnextv2_large, 8 epochs, 4 layers, cropped224, no posenc
# Key findings: 8 epochs optimal, moderate depth (2-4 layers) better than deep (8 layers)
num_epochs = {8}  # Analysis shows 8 epochs is optimal (vs 16 which overfits)
num_layers = {6}  # Testing 4-6 layers range
pos_unit = {60000}
dropouts = {0.3}  # Keep existing optimal dropout
# Focus on convnextv2 models, compare large vs tiny performance
img_enc_name = {
    'convnextv2_large',    # Best performer (0.517 avg F1)
    'convnextv2_tiny',     # Compare tiny model performance
}
resize_strategy = {'distorted'}  # Top 2 strategies from analysis
# positional encoding configuration - keep requested values
pos_length = {6000000}
pos_abs_th_front = {5}
pos_abs_th_end = {10}
pos_vec_coeff = {0, 0.3}  # 0=disabled, reduced positional weight from .5 to .3

# to see effect of training data size
block_guids_train = [
    batches.excluded_guids,  # full training set
    #  batches.excluded_guids 
    #  + batches.aapb_collaboration_27_a + batches.aapb_collaboration_27_b 
    #  + batches.aapb_collaboration_27_c + batches.aapb_collaboration_27_e,  # balanced sets only
    #  batches.excluded_guids 
    #  + batches.aapb_collaboration_27_bd01 + batches.aapb_collaboration_27_bd02 + batches.aapb_collaboration_27_bd03 
    #  + batches.aapb_collaboration_27_bd04 + batches.aapb_collaboration_27_bd05 + batches.aapb_collaboration_27_bd06,  # original sets only
    #  batches.excluded_guids 
    #  + batches.aapb_collaboration_27_a + batches.aapb_collaboration_27_b 
    #  + batches.aapb_collaboration_27_c + batches.aapb_collaboration_27_e + batches.aapb_collaboration_27_f 
    #  + batches.aapb_collaboration_27_bd01 + batches.aapb_collaboration_27_bd02 + batches.aapb_collaboration_27_bd03 
    #  + batches.aapb_collaboration_27_bd04 + batches.aapb_collaboration_27_bd05 # + batches.aapb_collaboration_27_bd06 
    ]
# since we now do validation on a fixed set, this parameter has no effect, keeping it for historical reasons
block_guids_valid = [batches.excluded_guids]

# "prebin" configurations.
# NOTE that postbin is not a part of the CV model, so is not handled here
# Including new human ambiguity-based binning schemes from issue #134
prebin = [
    #  'noprebin',  # Full granular classification (18 classes)
    'collapse-close', 
    #  'collapse-close-reduce-difficulty',
    #  'collapse-close-bin-lower-thirds',
    #  'ignore-difficulties'
]

clss_param_keys = ['num_epochs', 'num_layers', 'pos_length', 'pos_unit', 'dropouts', 
                   'img_enc_name', 'resize_strategy',
                   'pos_abs_th_front', 'pos_abs_th_end', 'pos_vec_coeff', 
                   'block_guids_train', 'block_guids_valid', 
                   'prebin']

## TF stitching grid search (for future)
tfMinTPScores = set()
tfMinTFScores = set()
tfLabelMapFns = set()
tfMinNegTFDurations = set()
tfMinTFDurations = set()
tfAllowOverlaps = set()

stit_param_keys = [
    "tfMinTPScores", "tfMinTFScores", "tfMinTFDurations", 
    # "tfAllowOverlaps"  # we don't have a proper evaluator for overlapping TFs
]

l = locals()


def get_classifier_training_grids():
    for vals in itertools.product(*[l[key] for key in clss_param_keys]):
        yield dict(zip(clss_param_keys, vals))

if __name__ == '__main__':
    import json
    grids = list(get_classifier_training_grids())
    # block_guids_train and block_guids_valid contain lists, which are just too noisy to print all 
    for grid in grids:
        grid['block_guids_train'] = f"<{len(grid['block_guids_train'])} GUIDs>"
        grid['block_guids_valid'] = f"<{len(grid['block_guids_valid'])} GUIDs>"
    print(json.dumps(grids, indent=2))
    for key in clss_param_keys:
        print(f"{key}: {len(l[key])} options")
    print(f"Total classifier training grid search configurations: {len(grids)}")
