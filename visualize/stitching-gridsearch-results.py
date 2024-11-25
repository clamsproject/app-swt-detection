"""
Script to generate hiplot-based parallel coordinates plots from the results of a grid search on the stitcher app.

"""
import csv
import json
import pathlib
import sys
from collections import defaultdict

import hiplot as hip

import modeling.config.bins

#  from modeling import gridsearch

hyperparams = {'tfMinTPScore': hip.ValueDef(value_type=hip.ValueType.NUMERIC, colormap='interpolateTurbo'),
               'tfMinTFScore': hip.ValueDef(value_type=hip.ValueType.NUMERIC, colormap='interpolateTurbo'),
               'tfLabelMapFn': hip.ValueDef(value_type=hip.ValueType.CATEGORICAL),
               'tfMinNegTFDuration': hip.ValueDef(value_type=hip.ValueType.NUMERIC, colormap='interpolateTurbo'),
               'tfMinTFDuration': hip.ValueDef(value_type=hip.ValueType.NUMERIC, colormap='interpolateTurbo'),
               'tfAllowOverlap': hip.ValueDef(value_type=hip.ValueType.CATEGORICAL)}
results_dir = pathlib.Path(sys.argv[1])

all_lbl = 'AVG'
all_binned_lbl = 'AVGBIN'
# only used when stitcher's labelMap is a identity function (no postbin)
binname = sys.argv[2] if len(sys.argv) > 2 else '!'

bins_of_interest = modeling.config.bins.binning_schemes.get(binname, {})
raw_lbls = "BSWLOMINEPYKUGTFCR"
lbls = [all_lbl] + list(raw_lbls) + list(bins_of_interest.keys())
inverse_bins = {v: k for k, vs in bins_of_interest.items() for v in vs}
data = defaultdict(list)


def is_identity(d):
    for key, value in d.items():
        if key != value:
            return False
    return True


for exp in results_dir.iterdir():
    if exp.is_dir() and (exp / 'results.csv').exists() and (exp / 'appConfiguration.json').exists():
        configs = json.load((exp / 'appConfiguration.json').open())
        # when there's actual postbinning, 
        if not is_identity(configs['tfLabelMap']):
            inverse_bins = configs['tfLabelMap']
            lbls = [all_lbl] + list(raw_lbls) + list(set(configs['tfLabelMap'].values()))
            bins_of_interest = defaultdict(list)
            for k, v in configs['tfLabelMap'].items():
                bins_of_interest[v].append(k)
        ## some "fixed" parameters
        # if configs['tfAllowOverlap']:
        #     continue
        # if configs['tfLabelMapFn'] == 'sum':
        #     continue
        # if 1 < configs['tfMinNegTFDuration'] < 1000:  # TP samplerate is 1000, so skip values under 
        #     continue

        base_params = {hp: configs[hp] for hp in hyperparams}

        exp_raw_scores = {}
        exp_bin_scores = defaultdict(lambda: {'P': {'filtered': [], 'stitched': []},
                                              'R': {'filtered': [], 'stitched': []},
                                              'F': {'filtered': [], 'stitched': []}})
        with (exp / 'results.csv').open() as f:
            reader = csv.DictReader(f)
            score_dict = {row['labels']: float(row['@@@ALL@@@']) for row in reader}
            for lbl in lbls:
                for met in 'PRF':
                    if f'{lbl} {met} STITCHED' in score_dict:
                        raw_score = score_dict.get(f'{lbl} {met} FILTERED', 0.0)
                        binarized_score = score_dict.get(f'{lbl} {met} STITCHED', 0.0)
                        exp_raw_scores[f'{lbl}-{met}-filtered'] = raw_score
                        exp_raw_scores[f'{lbl}-{met}-stitched'] = binarized_score
                        exp_raw_scores[f'{lbl}-{met}-diff'] = binarized_score - raw_score
                        exp_bin_scores[all_lbl][met]['filtered'].append(raw_score)
                        exp_bin_scores[all_lbl][met]['stitched'].append(binarized_score)
                        if lbl in inverse_bins:
                            exp_bin_scores[inverse_bins[lbl]][met]['filtered'].append(raw_score)
                            exp_bin_scores[inverse_bins[lbl]][met]['stitched'].append(binarized_score)
                            exp_bin_scores[all_binned_lbl][met]['filtered'].append(raw_score)
                            exp_bin_scores[all_binned_lbl][met]['stitched'].append(binarized_score)
        exp_bin_avg_scores = {}
        for lbl, scores in exp_bin_scores.items():
            for met, cond in scores.items():
                raw_avg = sum(cond['filtered']) / len(cond['filtered'])
                map_avg = sum(cond['stitched']) / len(cond['stitched'])
                diff_avg = map_avg - raw_avg
                exp_bin_avg_scores[f'{lbl}-{met}-filtered'] = raw_avg
                exp_bin_avg_scores[f'{lbl}-{met}-stitched'] = map_avg
                exp_bin_avg_scores[f'{lbl}-{met}-diff'] = diff_avg

        for d in [exp_raw_scores, exp_bin_avg_scores]:
            for lbl_and_type, score in d.items():
                # if score < 0:
                #     continue
                params = base_params.copy()
                l, m, c = lbl_and_type.rsplit('-', 2)
                t = f'{m}-{c}'
                params['score'] = score
                if l in bins_of_interest:
                    bin_num = list(bins_of_interest.keys()).index(l)
                    l_for_sorting = f'{bin_num}.{l}'
                elif l in inverse_bins:
                    bin_num = list(bins_of_interest.keys()).index(inverse_bins[l])
                    l_for_sorting = f'{bin_num}@{l}'
                else:
                    l_for_sorting = l
                    print(l, exp_bin_scores)
                    if len(bins_of_interest) != 0:  # meaning, a binning is used, then skip uninteresting labels
                        # the `|exp_bin_scores|==1` means the binning was actually a postbinning, in this case there's no "uninteresting" labels
                        if len(exp_bin_scores) > 1 and l != all_binned_lbl:
                            continue
                    else:
                        if l == all_lbl:  # when a binning is used, skip total avg, but keep bin-level avg
                            continue
                params['label'] = l_for_sorting
                data[t].append(params)

# print(data)

for k, v in data.items():
    out_fname = f'stitcher-results-{binname}-{k}.html'
    print(k, f'({len(v)} items)', '>>', out_fname)
    p = hip.Experiment.from_iterable(v)
    p.parameters_definition = hyperparams
    p.parameters_definition['score'] = hip.ValueDef(value_type=hip.ValueType.NUMERIC, colormap='interpolateTurbo')
    p.colorby = 'score'
    with open(out_fname, 'w') as out_f:
        p.to_html(out_f)
