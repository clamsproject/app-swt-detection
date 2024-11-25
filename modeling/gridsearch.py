import itertools
import math

import modeling.backbones
import modeling.config.bins
from modeling.config.batches import unintersting_guids, aapb_collaboration_27_b, aapb_collaboration_27_c, \
    aapb_collaboration_27_e


## TP classifier training grid search
# parameter values from the best performing models in v5.0
split_size = {math.inf}
num_epochs = {10}
num_layers = {4}
pos_unit = {60000}
dropouts = {0.3}
# img_enc_name = modeling.backbones.model_map.keys()
img_enc_name = {'convnext_lg', 'convnext_small', 'convnext_tiny'}

# positional encoding configuration best performed as of v6.0
pos_length = {6000000}
pos_abs_th_front = {5}
pos_abs_th_end = {10}
pos_vec_coeff = {0, 0.5}  # when 0, positional encoding is not enabled

# to see effect of training data size
block_guids_train = [
    
    # aapb_collaboration_27_a + aapb_collaboration_27_b + aapb_collaboration_27_c + aapb_collaboration_27_e,  # no training data
    ## 20 + 21 + 20 + 60 = 121 videos (excluding `d` batch) with 1 uninsteresting video and 40 videos in `pbd` subset in `e` batch
    # unintersting_guids + aapb_collaboration_27_b + aapb_collaboration_27_c + aapb_collaboration_27_e,  # only the first "dense" annotations (shown as 0101@xxx in the bar plotting from see_results.py )
    # unintersting_guids + aapb_collaboration_27_c + aapb_collaboration_27_e,  # adding "sparse" annotations (shown as 0061@xxx)
    # unintersting_guids + aapb_collaboration_27_e,  # adding the second "dense" annotations (shown as 0081@xxx)
    unintersting_guids,  # adding the "challenging" images, this is the "full" size (shown as 0001@xxx, but really using 80 guids from `a` + `b` + `c` + `bm`)
    # note that the "uninstresting" video is never used in all training sets
]
# since we now do validation on a fixed set, this parameter has no effect, keeping it for historical reasons
block_guids_valid = [
    aapb_collaboration_27_b + aapb_collaboration_27_e,  # block all loosely-annotated videos and the challenging images
    #  unintersting_guids,  # effectively no block except
]

# "prebin" configurations. 
# NOTE that postbin is not a part of the CV model, so is not handled here
# for single binning configuration, just use the binning dict
# for multiple binning configurations (for experimental reasons), use the binning scheme names (str)
prebin = ['noprebin']
# prebin = []

clss_param_keys = ['split_size', 'num_epochs', 'num_layers', 'pos_length', 'pos_unit', 'dropouts', 'img_enc_name', 
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

