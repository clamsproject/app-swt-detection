import csv
import pathlib
import sys
from collections import defaultdict

import hiplot as hip
import yaml

import modeling.config.bins
import modeling.gridsearch

hyperparams = modeling.gridsearch.clss_param_keys

results_dir = pathlib.Path(sys.argv[1])

avg_alllbl = '!AVG'
avg_allbin = '!AVGBIN'

labels_with_isseus = [
    'U',  # no training instances
    'K',  # no evaluation instances
    'W',  # almost no instances both in training and evaluation
]

bin_scheme_name = sys.argv[2] if len(sys.argv) > 2 else 'nomap'
bins_of_interest = modeling.config.bins.binning_schemes.get(bin_scheme_name, {})
for k, vs in bins_of_interest.items():
    no_subtypes = [v for v in vs if ':' not in v]
    bins_of_interest[k] = no_subtypes


def backbone_sorter():
    import re
    backbone_scr = pathlib.Path(modeling.gridsearch.__file__).parent / 'backbones.py'
    backbone_names = []
    with open(backbone_scr) as backbone_f:
        for line in backbone_f:
            # regex match for `    name = "vgg16"` or `    name = 'bn_vgg16'`
            match = re.match(r'\s+name\s*=\s*["\']([^"\']+)["\']', line)
            if match:
                backbone_names.append(match.group(1))
    return backbone_names
        
raw_lbls = modeling.FRAME_TYPES 
lbls = [avg_alllbl] + raw_lbls + list(bins_of_interest.keys())
inverse_bins = {v: k for k, vs in bins_of_interest.items() for v in vs}
data = defaultdict(list)


def is_identity(d):
    for key, value in d.items():
        if key != value:
            return False
    return True


experiments = defaultdict(set)
# training results are stored as <TIMESTAMP>.<BACKBONE_NAME>.<POSENC>.{csv,yml}
for f in results_dir.iterdir():
    exp_id, ext = f.name.rsplit('.', 1)
    experiments[exp_id].add(ext)

exps = [exp_id for exp_id, exts in experiments.items() if 'csv' in exts and 'yml' in exts]


def clean_config(config, prebin_name=None):
    """
    Clean up the configuration found in a yml file with more human friendly names. 
    """
    bgtrain = list(set(config['block_guids_train']))
    bgvalid = list(set(config['block_guids_valid']))
    config['block_guids_train'] = f'{len(bgtrain):04}@{hash(str(sorted(bgtrain)))}'
    config['block_guids_valid'] = f'{len(bgvalid):04}@{hash(str(sorted(bgvalid)))}'

    # a short string name of the prebin can be passed as an argument or can be generated from dictionary in the config 
    if prebin_name:
        config['prebin'] = prebin_name
    elif 'prebin' in config:
        config['prebin'] = f'{len(config["prebin"])}way@{hash(str(config["prebin"]))}'
    else:
        config['prebin'] = 'None'

    # Keep pos_vec_coeff as numeric value instead of binary
    # (no deletion, keep the actual coefficient)

    # del config['split_size']
    return config

img_encer_sorter = backbone_sorter()
for exp in exps:
    configs = yaml.safe_load((results_dir / f'{exp}.yml').open())
    # TODO (krim @ 11/6/24): add handling of prebins when we're using it (currently not using)
    # if 'prebin' in configs:
    configs = clean_config(configs)
    
    # skip uninsteresting configurations
    if configs['pos_unit'] == 1000:
        continue
    

    base_params = {hp: configs[hp] for hp in hyperparams if hp in configs}
    # then add back all configs keys that were renamed in `clean_config`
    for k, v in configs.items():
        if k not in base_params:
            base_params[k] = v

    exp_raw_scores = defaultdict(lambda: {'P': 0.0, 'R': 0.0, 'F': 0.0})
    exp_bin_scores = defaultdict(lambda: {'P': [], 'R': [], 'F': []})
    
    with (results_dir / f'{exp}.csv').open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'Confusion Matrix' in row['Model_Name'] or not row:
                break
            lbl = row['Label']
            if lbl == 'Overall':
                lbl = avg_alllbl
            for met in 'Precision Recall F1-Score'.split():
                exp_raw_scores[lbl][met[0]] = float(row[met])
                if lbl in inverse_bins:
                    exp_bin_scores[inverse_bins[lbl]][met[0]].append(float(row[met]))
                    exp_bin_scores[avg_allbin][met[0]].append(float(row[met]))
    for binname, scores in exp_bin_scores.items():
        for met, scorelist in scores.items():
            exp_raw_scores[binname][met] = sum(scorelist) / len(scorelist)
            
    for lbl, scores in exp_raw_scores.items():
        if lbl in labels_with_isseus:
            continue
        for met, score in scores.items():
            params = base_params.copy()
            params['score'] = score
            # when there's any bin (post bin or average bin), we prefix bin name
            if lbl in bins_of_interest:
                bin_num = list(bins_of_interest.keys()).index(lbl)
                l_for_sorting = f'{bin_num}.{lbl}({"".join(bins_of_interest[lbl])})'
            # AND label as well
            elif lbl in inverse_bins:
                # except for that the bin is a singleton
                if bins_of_interest[inverse_bins[lbl]] == [lbl]:
                    continue
                bin_num = list(bins_of_interest.keys()).index(inverse_bins[lbl])
                l_for_sorting = f'{bin_num}@{lbl}'
            # if the label wasn't in the bin dicts, it's an "uninteresting" one
            else:
                l_for_sorting = lbl
                # if average binning is used, 
                if len(bins_of_interest) != 0:  
                    # then skip uninteresting labels
                    if lbl != avg_allbin:
                        continue
                # when "binary" binning, skip the average
                if len(bins_of_interest) == 1 and lbl == avg_allbin:
                    continue
            # else:
                #     if lbl == avg_alllbl:  # when a binning is used, skip total avg, but keep bin-level avg
                #         continue
            params['label'] = l_for_sorting
            params['img_enc_name'] = f'{img_encer_sorter.index(params["img_enc_name"]):03}.{params["img_enc_name"]}'
            
            # remove unused params
            for unused in ('pos_unit', 'pos_length', 'pos_abs_th_end', 'pos_abs_th_front', 'prebin'):
                params.pop(unused)
            
            data[met].append(params)

for k, v in data.items():
    out_html_fname = f'{results_dir.name}-gridsearch-results-{bin_scheme_name}-{k}.html'
    out_csv_fname = f'{results_dir.name}-gridsearch-results-{bin_scheme_name}-{k}.csv'
    print(k, f'({len(v)} items)', '>>', out_html_fname)
    p = hip.Experiment.from_iterable(v)
    # p.parameters_definition = hyperparams
    for hp in hyperparams:
        if hp == 'score' or hp.startswith('num_') or hp in ('pos_vec_coeff', 'dropouts'):
            p.parameters_definition[hp] = hip.ValueDef(value_type=hip.ValueType.NUMERIC, colormap='interpolateTurbo')
        elif hp in ('img_enc_name', 'block_guids_train', 'block_guids_valid', 'prebin'):
            p.parameters_definition[hp] = hip.ValueDef(value_type=hip.ValueType.CATEGORICAL, colormap='interpolateViridis')
    p.colorby = 'score'
    with open(out_html_fname, 'w') as out_f:
        p.to_html(out_f)
    print(k, f'({len(v)} items)', '>>', out_csv_fname)
    with open(out_csv_fname, 'w') as out_f:
        p.to_csv(out_f)
