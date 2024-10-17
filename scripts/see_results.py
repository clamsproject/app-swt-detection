import argparse
import base64
import csv
import os
import pathlib
from collections import defaultdict
from io import BytesIO
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import yaml


def process_kfold_validation_results(directory):
    """
    THIS FUNCTION IS OUTDATED since we no longer actively use k-fold validation.
    Hence, the code is not compatible with new file naming convention and structure for new "fixed" validateion experiment results.
    
    1. Iterate over all files in the directory
    2. Get configuration information
    3. Calculate the averages of accuracy, precision, recall, and f1-score for each label for each set of k_fold results.
    4. Save and return them in a dictionary format.
    :param directory: where evaluation results files are stored
    :return: 1. A dictionary with ids as keys and the configuration dictionary as values : dict[id][parameter]->value
            2. A dictionary with ids as keys and the macro average dictionary as values: dict[id][label][metric]->value
    """
    result_sets = defaultdict(list)
    # Iterate over all files in the directory
    if directory == "":
        directory = os.getcwd()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        result_sets[filename.split(".")[0]].append(file_path)

    # Store the evaluation results in the dictionary form
    macro_avgs = {}
    configs = {}
    for key, value in result_sets.items():
        macro_avg = defaultdict(lambda: defaultdict(float))
        i = 0
        for file in value:
            if file.endswith(".csv"):
                i += 1
                with open(file, "r") as f:
                   csv_reader = csv.DictReader(f)
                   for row in csv_reader:
                       macro_avg[row['Label']]['Accuracy'] += float(row['Accuracy'])
                       macro_avg[row['Label']]['Precision'] += float(row['Precision'])
                       macro_avg[row['Label']]['Recall'] += float(row['Recall'])
                       macro_avg[row['Label']]['F1-Score'] += float(row['F1-Score'])

            if file.endswith(".yml"):
                with open(file, "r") as f:
                    data = yaml.safe_load(f)
                # delete unnecessary items
                configs[key] = clean_config(data)

        # Calculate macro averages
        for k, v in macro_avg.items():
            for metric in v:
                v[metric] = v[metric]/float(i)

        # Add overall macro averages for all labels for each set.
        num_classes = len(macro_avg)
        macro_avg["overall"] = defaultdict(float)
        for k, v in macro_avg.items():
            if k != "overall":
                for metric in v:
                    macro_avg["overall"][metric] += v[metric]/num_classes

        macro_avgs[key] = macro_avg

    return configs, macro_avgs


def clean_config(config, prebin_name=None):
    """
    Clean up the configuration found in a yml file with more human friendly names. 
    """
    config['block_guids_train'] = f'{len(config["block_guids_train"])}@{hash(str(sorted(config["block_guids_train"])))}'
    config['block_guids_valid'] = f'{len(config["block_guids_valid"])}@{hash(str(sorted(config["block_guids_valid"])))}'
    
    # a short string name of the prebin can be passed as an argument or can be generated from dictionary in the config 
    if prebin_name:
        config['prebin'] = prebin_name
    elif 'prebin' in config:
        config['prebin'] = f'{len(config["prebin"])}way@{hash(str(config["prebin"]))}'
    else:
        config['prebin'] = 'None'
        
    config['posenc'] = config['pos_vec_coeff'] > 0
    del config['pos_vec_coeff']
    
    del config['split_size']
    return config


def process_fixed_validation_results(directory, include_negative_label=False):
    configs = {}
    scores = {}
    for csv_fname in pathlib.Path(directory).glob('*.csv'):
        key = csv_fname.stem
        timestamp, bb_name, bin_name, posenc = key.split('.')
        posenc = posenc[-1] == 'T'
        score = defaultdict(lambda: defaultdict(float))
        with open(csv_fname, "r") as csv_f:
            csv_reader = csv.DictReader(csv_f)
            for row in csv_reader:
                if 'Confusion Matrix' in row['Model_Name'] or not row:
                    break
                # ignore negative class
                if row['Label'] == '-' and not include_negative_label:
                    continue
                score[row['Label']]['Accuracy'] += float(row['Accuracy'])
                score[row['Label']]['Precision'] += float(row['Precision'])
                score[row['Label']]['Recall'] += float(row['Recall'])
                score[row['Label']]['F1-Score'] += float(row['F1-Score'])
        config_fname = csv_fname.with_suffix('.yml')
        with open(config_fname, "r") as yml_f:
            config = yaml.safe_load(yml_f)
            config = clean_config(config, bin_name)
        # delete unnecessary items
            configs[key] = config
        scores[key] = score
    return configs, scores


def get_inverse_configs(configs):
    """
    Get inverse dictionary for configurations that allow user to find IDs from configurations.
    :param configs: A dictionary with IDs as keys and a dictionary with configurations as values.
    :return: A nested dictionary with parameter name as 1st keys, parameter value as 2nd key and a set of IDs as values.
    """
    inverse_dict = defaultdict(lambda: defaultdict(set))
    for key, val in configs.items():
        for k, v in val.items():
            inverse_dict[k][v].add(key)

    return inverse_dict


def get_grid(configs):
    """
    Get grid of configurations used.
    :param configs: A dictionary with IDs as keys and a dictionary with configurations as values.
    :return: A dictionary with parameter name as keys and list of parameter used in grid search as value.
    """
    grid = defaultdict(set)
    for value in configs.values():
        for k, v in value.items():
            grid[k].add(v)

    refined_grid = {}
    for key, val in grid.items():
        if len(val) > 1:
            refined_grid[key] = list(val)

    return refined_grid


def get_labels(macroavgs):
    """
    Get list of labels. This is needed because some sets doesn't have results for some labels.
    :param macroavgs: A dictionary of macro averages of results that was got from get_configs_and_macroavgs function.
    :return: A list of labels
    """
    labels = set()
    for key, val in macroavgs.items():
        labels.update(val.keys())
    if '-' in labels:
        labels.remove('-')
    return list(labels)


def plot_bar_graphs(axis, exp_group, score_dict, config_dict, target_label, target_var, var_vals, colorscheme):
    # For each pair, form a data dictionary as data = { ID1: [accuracy, precision, recall, f1], ...}
    # and plot a bar graph
    # re-order the pair to show the variable values in the same order as in the grid
    ordered_group = [None] * len(var_vals)
    for i, value in enumerate(var_vals):
        for exp_id in exp_group:
            if config_dict[exp_id][target_var] == value:
                ordered_group[i] = exp_id
    metrics = score_dict[ordered_group[0]]["Overall"].keys()
    data = defaultdict(list)
    metric_list = [f'Avg {m}' for m in metrics]
    for i, exp_id in enumerate(ordered_group):
        label_found = False
        for l in score_dict[exp_id].keys():
            if l.startswith(target_label):
                for metric in metrics:
                    data[exp_id].append(score_dict[exp_id][l][metric])
                label_found = True
                break
        if not label_found:
            data[exp_id].append(0.0)
    data = dict(data)

    if len(data) == 0:
        return None, None
    # plot a bar graph
    x = np.arange(len(metric_list))  # the label locations
    width = 1/(len(data)+1)  # the width of the bars
    multiplier = 0

    for exp_id, scores in data.items():
        if len(scores) == 1 and scores[0] == 0.0:
            continue
        id_variable = str(target_var) + ": " + str(config_dict[exp_id][target_var])
        offset = width * multiplier
        rects = axis.bar(x + offset, scores, width, label=id_variable, color=colorscheme[str(config_dict[exp_id][target_var])])
        axis.bar_label(rects, fmt='%.6s', fontsize='small', rotation='vertical', padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    axis.set_ylabel('Score')
    axis.set_title(str(target_label))
    axis.set_xticks(x + width * (len(data) - 1) / 2, metric_list)
    axis.legend(loc='center left', fontsize='small', ncol=1, bbox_to_anchor=(1, 0.5))
    axis.set_ylim(0.0, 1.15)
    # Show information on fixed parameters.
    string_configs = f'{exp_id}\n'
    for k, v in config_dict[exp_id].items():
        if k != target_var:
            string_configs += str(k) + ": " + str(v) + "\n"
    axis.text(0.99, 0.97, string_configs,
              verticalalignment='bottom', horizontalalignment='right',
              transform=axis.transAxes,
              color='green', fontsize='small')
    return axis


def get_pairs_to_compare(grid, inverse_configs, variable):
    """
    Get a list of pairs(lists of IDs) where all configurations are the same except for one given variable.
    :param grid: Grid of configurations used in this experiment
    :param inverse_configs: A dictionary that allows user to search IDs from configurations.
    :param variable: the variable parameter.
    :return:  A list of pairs(lists of IDs)
    """

    # Delete variable key from grid and inverse_configs dictionary
    del grid[variable]
    del inverse_configs[variable]
    # Form all possible configurations of parameters from grid and store it as a list of dictionary form.
    conf_dicts = [dict(zip(grid.keys(), config)) for config in list(product(*grid.values()))]

    # Get all the possible lists of exps using inverse_configs dictionary and intersection of them for every config
    pair_list = []
    for conf_dict in conf_dicts:
        list_of_sets = [inverse_configs[param_name][val] for param_name, val in conf_dict.items()]

        # Get intersection of sets of IDs for given configurations
        intersection_result = list_of_sets[0]
        # Iterate over the remaining sets and find the intersection
        for s in list_of_sets[1:]:
            intersection_result = intersection_result.intersection(s)

        if len(intersection_result) > 0:
            pair_list.append(list(intersection_result))

    return pair_list


def compare_pairs(exp_groups, scores, conf_grid, configs, target_lbl, target_var, var_vals, interactive_plots=True):
    """
    For list of pairs got from get_pairs_to_compare function, compare each pair by plotting bar graphs for given label.
    :param exp_groups: got from get_pairs_to_compare function for given variable
    :param scores: PRF scores from each experiment configuration
    :param conf_grid: grid of configurations used in this experiment
    :param configs: actual configurations used in this experiment
    :param target_lbl: User choice of label (including overall) to show scores in the graph. 
                       a special value `all` will generate plots for all "normalized" labels (put them horizontally)
    :param target_var: configuration key name to use as a variable to compare, all other keys are fixed.
    :param var_vals: list of values for the variable to compare
    :param interactive_plots: flag to show plots in realtime. If false, the program will save all the plots in a html
    """

    # Form parameter to color dictionary for consistency in color across all pairs
    param_to_color = dict((str(value), f'C{i}') for i, value in enumerate(conf_grid[target_var]))

    html = '<html><head><title>Comparison of pairs</title></head><body>'

    # For each pair, form a data dictionary as data = { ID1: [accuracy, precision, recall, f1], ...}
    # and plot a bar graph
    for group in exp_groups:
        if target_lbl == 'all':
            interested_lbls = "Ba Sl Ch Cr".split()
            fig, axes = plt.subplots(1, len(interested_lbls), figsize=(45, 5), sharex=True, sharey=True)
            plt.subplots_adjust(wspace=1)
            for ax, lbl in zip(np.ravel(axes), interested_lbls):
                plot_bar_graphs(ax, group, scores, configs, lbl, target_var, var_vals, param_to_color)
        else:
            fig, ax = plt.subplots()
            plot_bar_graphs(ax, group, scores, configs, target_lbl, target_var, var_vals, param_to_color)

        if interactive_plots:
            plt.show()
        else:
            temp_io_stream = BytesIO()
            fig.savefig(temp_io_stream, format='png', bbox_inches='tight')
            htmlized = f'<p><img src="data:image/png;base64,{base64.b64encode(temp_io_stream.getvalue()).decode("utf-8")}"></p>'
            html += htmlized
        plt.cla()

    if not interactive_plots:
        html += '</body></html>'
        with open(f'results-comparison-{target_var}-{target_lbl}.html', 'w') as f:
            f.write(html)


def user_input_variable(grid):
    """
    A function to receive user input on which parameter to vary.
    :param grid: dictionary of variable names and list of values.
    :return: user choice among parameter names in grid.
    """
    try:
        choice = str(input("\nEnter one parameter to vary from " + str(list(grid.keys())) + "\n:"))
        if choice in grid.keys():
            return choice
        else:
            raise ValueError("Invalid argument for variable. Please enter one of ", list(grid.keys()))
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid argument for variable. Please enter one of ", list(grid.keys()))


def user_input_label(label_list):
    """
    A function to receive user input on which label to plot and show the scores.
    :param label_list:
    :return: user choice among label names in label_list.
    """
    try:
        choice = str(input("\nEnter a label for comparing results: " + str(label_list) + "\n:"))
        if choice in label_list:
            return choice
        else:
            raise ValueError("Invalid argument for variable. Please enter one of ", label_list)
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid argument for variable. Please enter one of ",
                                         label_list)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        type=str,
        help="Directory with result and configuration files",
        default="",
    )
    parser.add_argument(
        '-l', '--label',
        default='overall',
        action='store',
        nargs='?',
        help='Pick a label to compare, default is overall, meaning all labels are plotted'
    )
    parser.add_argument(
        '-k', '--config-key',
        default=None,
        action='store',
        nargs='?',
        help='Pick a config key to "pin" the comparison to. '
             'When this is not set, the program runs in "interactive" mode, '
             'where the user is prompted to pick a key and a label to compare.'
    )
    parser.add_argument(
        '-i', '--interactive-plots',
        action='store_true',
        help='Flag to show plots in interactive mode. If not set, the program will save all the plots in a html file.'
    )
    parser.add_argument(
        '-n', '--negativelabel', 
        action='store_true',
        help='Flag to include the negative label when averaging scores.'
    )

    args = parser.parse_args()

    # Get necessary dictionaries and lists for processing the comparison.
    is_kfold = bool(any(pathlib.Path(args.directory).glob("*kfold*.csv")))
    if is_kfold:
        configs, macroavgs = process_kfold_validation_results(args.directory)
    else:
        configs, macroavgs = process_fixed_validation_results(args.directory, args.negativelabel)
    label_list = get_labels(macroavgs)
    label_list.append('all')
    inverse_configs = get_inverse_configs(configs)
    grid = get_grid(configs)
    if args.config_key is None:
        # Get user inputs and prepare the list of pairs
        choice_variable = user_input_variable(grid)
        choice_label = user_input_label(label_list)
    else:
        if args.config_key in grid:
            choice_variable = args.config_key
        else:
            raise argparse.ArgumentTypeError("Invalid argument for variable. Please enter one of ", list(grid.keys()))
        if args.label in label_list:
            choice_label = args.label
        else:
            raise argparse.ArgumentTypeError("Invalid argument for label. Please enter one of ", label_list)
    variable_values = sorted(grid[choice_variable].copy())
    list_of_pairs = get_pairs_to_compare(grid.copy(), inverse_configs, choice_variable)
    # Show the comparison results of pairs in bar graphs
    compare_pairs(list_of_pairs, macroavgs, grid, configs.copy(), choice_label, choice_variable, variable_values,
                  interactive_plots=args.interactive_plots)
