import itertools
import math

import modeling.backbones
import modeling.config.bins
from modeling.config.batches import unintersting_guids, aapb_collaboration_27_b, aapb_collaboration_27_c, aapb_collaboration_27_e, guids_with_confirmed_slates_images


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
    
    # aapb_collaboration_27_a + aapb_collaboration_27_b + aapb_collaboration_27_c + aapb_collaboration_27_e + guids_with_confirmed_slates_images,  # exclude all training data
    # First, count of GUIDs in annotation batches: a + b + c + e + f(?) = 20 + 21 + 20 + 60 + 1118 = 1239 videos 
    # excluding `d` batch (reserved for evaluation),  including 1 uninsteresting video in `27-b` batch 
    # (`d` batch has a single GUID, and never "vectorized" to `npy` files, hence not explicitly specified here)
    # unintersting_guids,  # this used to be the "full" size (hashed as 0001@xxx in the result visualization until v7.5)
    unintersting_guids + guids_with_confirmed_slates_images,  # OLD v7.1rc training set (hashed as 1119@xxx), now this is the "full" training set that we decided to hold "confirmed-slate" set out for evaluation of KIE task (see https://github.com/clamsproject/aapb-collaboration/issues/27#issuecomment-2841848595)
    # note that the "uninstresting" video is never used in all training sets
    # also note that when `split_size` is very large, no k-fold is performed, and the `pbd` set is used for fixed validation
]
# since we now do validation on a fixed set, this parameter has no effect, keeping it for historical reasons
block_guids_valid = [
    unintersting_guids + aapb_collaboration_27_b + aapb_collaboration_27_e + guids_with_confirmed_slates_images,  # block all loosely-annotated videos and the challenging images
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

