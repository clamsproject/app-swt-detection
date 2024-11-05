import itertools
import math

import modeling.backbones
import modeling.bins

# parameter values from the best performing models in v5.0
split_size = {math.inf}
num_epochs = {10}
num_layers = {4}
pos_unit = {60000}
pos_enc_dim = {256}
dropouts = {0.1}
# img_enc_name = modeling.backbones.model_map.keys()
img_enc_name = {'convnext_lg', 'convnext_base', 'convnext_small', 'convnext_tiny'}

# positional encoding configuration best performed as of v6.0
pos_length = {6000000}
pos_abs_th_front = {5}
pos_abs_th_end = {10}
pos_vec_coeff = {0, 0.5}  # when 0, positional encoding is not enabled

# training batches see https://github.com/clamsproject/aapb-annotations/tree/main/batches for more details
unintersting_guids = ["cpb-aacip-254-75r7szdz"]   # the most "uninteresting" video (88/882 frames annotated)
aapb_collaboration_27_a = [
    "cpb-aacip-129-88qc000k",
    "cpb-aacip-f2c34dd1cd4",
    "cpb-aacip-191-40ksn47s",
    "cpb-aacip-507-028pc2tp2z",
    "cpb-aacip-507-0k26970f2d",
    "cpb-aacip-507-0z70v8b17g",
    "cpb-aacip-512-542j67b12n",
    "cpb-aacip-394-149p8fcw",
    "cpb-aacip-08fb0e1f287",
    "cpb-aacip-512-t43hx1753b",
    "cpb-aacip-d0f2569e145",
    "cpb-aacip-d8ebafee30e",
    "cpb-aacip-c72fd5cbadc",
    "cpb-aacip-b6a2a39b7eb",
    "cpb-aacip-512-4b2x34nv4t",
    "cpb-aacip-512-416sx65d21",
    "cpb-aacip-512-3f4kk95f7h",
    "cpb-aacip-512-348gf0nn4f",
    "cpb-aacip-516-cc0tq5s94c",
    "cpb-aacip-516-8c9r20sq57",
]
aapb_collaboration_27_b = [
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
aapb_collaboration_27_c = [
    "cpb-aacip-0d338c39a45",
    "cpb-aacip-0acac5e9db7",
    "cpb-aacip-0bdc7c8ecc5",
    "cpb-aacip-1032b1787b4",
    "cpb-aacip-516-qf8jd4qq96",
    "cpb-aacip-259-kh0dzd78",
    "cpb-aacip-259-nc5sb374",
    "cpb-aacip-259-mw28cq94",
    "cpb-aacip-259-mc8rg22j",
    "cpb-aacip-259-5717pw8g",
    "cpb-aacip-259-pr7msz5c",
    "cpb-aacip-259-g737390m",
    "cpb-aacip-259-pc2t780t",
    "cpb-aacip-259-q814r90k",
    "cpb-aacip-259-cz325478",
    "cpb-aacip-259-vh5cgj9t",
    "cpb-aacip-259-gt5ff704",
    "cpb-aacip-259-gx44t714",
    "cpb-aacip-259-pr7msz3w",
    "cpb-aacip-259-zg6g5589",
]
aapb_collaboration_27_d = [
    "cpb-aacip-259-wh2dcb8p"
]  # this is kept for evaluation set, should not be used for training!!!
# new image-level annotation added after v6.1
# "challenging images" from later annotation (`bm` set and `pbd` set, 60 videos, 2024 summer)
# recorded as `aapb-collaboration-27-e` in the annotation repo
guids_with_challenging_images_bm = [
    "cpb-aacip-00a9ed7f2ba",
    "cpb-aacip-0ace30f582d", 
    "cpb-aacip-0ae98c2c4b2",
    "cpb-aacip-0b0c0afdb11",
    "cpb-aacip-0bb992d2e7f",
    "cpb-aacip-0c0374c6c55",
    "cpb-aacip-0c727d4cac3",
    "cpb-aacip-0c74795718b",
    "cpb-aacip-0cb2aebaeba",
    "cpb-aacip-0d74af419eb",
    "cpb-aacip-0dbb0610457",
    "cpb-aacip-0dfbaaec869",
    "cpb-aacip-0e2dc840bc6",
    "cpb-aacip-0ed7e315160",
    "cpb-aacip-0f3879e2f22",
    "cpb-aacip-0f80359ada5",
    "cpb-aacip-0f80a4f5ed2",
    "cpb-aacip-0fe3e4311e1",
    "cpb-aacip-1a365705273",
    "cpb-aacip-1b295839145",
]
guids_with_challenging_images_pbd = [
    "cpb-aacip-110-16c2ftdq",
    "cpb-aacip-120-1615dwkg",
    "cpb-aacip-120-203xsm67",
    "cpb-aacip-15-70msck27",
    "cpb-aacip-16-19s1rw84",
    "cpb-aacip-17-07tmq941",
    "cpb-aacip-17-58bg87rx",
    "cpb-aacip-17-65v6xv27",
    "cpb-aacip-17-81jhbz0g",
    "cpb-aacip-29-61djhjcx",
    "cpb-aacip-29-8380gksn",
    "cpb-aacip-41-322bvxmn",
    "cpb-aacip-41-42n5tj3d",
    "cpb-aacip-110-35gb5r94",
    "cpb-aacip-111-655dvd99",
    "cpb-aacip-120-19s1rrsp",
    "cpb-aacip-120-31qfv097",
    "cpb-aacip-120-73pvmn2q",
    "cpb-aacip-120-80ht7h8d",
    "cpb-aacip-120-8279d01c",
    "cpb-aacip-120-83xsjcb2",
    "cpb-aacip-17-88qc0md1",
    "cpb-aacip-35-36tx99h9",
    "cpb-aacip-42-78tb31b1",
    "cpb-aacip-52-84zgn1wb",
    "cpb-aacip-52-87pnw5t0",
    "cpb-aacip-55-84mkmvwx",
    "cpb-aacip-75-13905w9q",
    "cpb-aacip-75-54xgxnzg",
    "cpb-aacip-77-02q5807j",
    "cpb-aacip-77-074tnfhr",
    "cpb-aacip-77-1937qsxt",
    "cpb-aacip-77-214mx491",
    "cpb-aacip-77-24jm6zc8",
    "cpb-aacip-77-35t77b2v",
    "cpb-aacip-77-44bp0mdh",
    "cpb-aacip-77-49t1h3fv",
    "cpb-aacip-77-81jhbv89",
    "cpb-aacip-83-074tmx7h",
    "cpb-aacip-83-23612txx",
]
aapb_collaboration_27_e = guids_with_challenging_images_bm + guids_with_challenging_images_pbd
# this `pbd` subset contains 40 videos with 15328 (non-transitional) + 557 (transitional) = 15885 frames
# then updated with more annotations 19331 (non-transitional) + 801 (transitional) = 20132 frames
# we decided to use this subset for the fixed validation set (#116)
guids_for_fixed_validation_set = guids_with_challenging_images_pbd

# to see effect of training data size
block_guids_train = [
    
    # aapb_collaboration_27_a + aapb_collaboration_27_b + aapb_collaboration_27_c + aapb_collaboration_27_e,  # no training data
    ## 20 + 21 + 20 + 60 = 121 videos (excluding `d` batch) with 1 uninsteresting video and 40 videos in `pbd` subset in `e` batch
    unintersting_guids + aapb_collaboration_27_b + aapb_collaboration_27_c + aapb_collaboration_27_e,  # only the first "dense" annotations (shown as 0101@xxx in the bar plotting from see_results.py )
    unintersting_guids + aapb_collaboration_27_c + aapb_collaboration_27_e,  # adding "sparse" annotations (shown as 0061@xxx)
    unintersting_guids + aapb_collaboration_27_e,  # adding the second "dense" annotations (shown as 0081@xxx)
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
prebin = [modeling.bins.nobinning]

param_keys = ['split_size', 'num_epochs', 'num_layers', 'pos_length', 'pos_unit', 'dropouts', 'img_enc_name', 'pos_abs_th_front', 'pos_abs_th_end', 'pos_vec_coeff', 'block_guids_train', 'block_guids_valid', 'prebin']
l = locals()
configs = []
for vals in itertools.product(*[l[key] for key in param_keys]):
    configs.append(dict(zip(param_keys, vals)))
