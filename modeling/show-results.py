import os
from collections import defaultdict
import csv
import yaml
import argparse
from itertools import product
import matplotlib.pyplot as plt
import numpy as np

# list of bins
# Since the bins parameters are too long to print or show on the plot, they are numbered by index.
bins = [
    {'pre': {'bars': ['B'], 'slate': ['S', 'S:H', 'S:C', 'S:D', 'S:G'], 'other-opening': ['W', 'L', 'O', 'M'],
             'chyron': ['I', 'N', 'Y'], 'not-chyron': ['P', 'K', 'G', 'T', 'F'], 'credits': ['C'], 'copyright': ['R']},
     'post': {'bars': ['bars'], 'slate': ['slate'], 'chyron': ['chyron'], 'credits': ['credits']}},
    {'post': {'bars': ['B'], 'slate': ['S', 'S:H', 'S:C', 'S:D', 'S:G'], 'chyron': ['I', 'N', 'Y'], 'credits': ['C']}},


    {'pre': {'bars': ['B'], 'slate': ['S', 'S:H', 'S:C', 'S:D', 'S:G'], 'warning': ['W'], 'opening': ['O'],
             'main_title': ['M'], 'chyron': ['I'], 'credits': ['C'], 'copyright': ['R']},
     'post': {'bars': ['bars'], 'slate': ['slate'], 'chyron': ['chyron'], 'credits': ['credits']}},
    {'post': {'bars': ['B'], 'slate': ['S', 'S:H', 'S:C', 'S:D', 'S:G'], 'chyron': ['I'], 'credits': ['C']}},


    {'pre': {'chyron': ['I', 'N', 'Y'], 'person-not-chyron': ['E', 'P', 'K']}, 'post': {'chyron': ['chyron']}},
    {'post': {'chyron': ['I', 'N', 'Y']}},
]


def get_configs_and_macroavgs(directory):
    """
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
                data['bins'] = bins.index(data['bins'])  # set bin config as an index of the bin
                # delete unnecessary items
                del data['block_guids_train']
                del data['block_guids_valid']
                del data['num_splits']
                configs[key] = data

        # Calculate macro averages
        for k, v in macro_avg.items():
            for metric in v:
                v[metric] = v[metric]/float(i)

        # Add overall macro averages for all labels for each set.
        num_classes = len(macro_avg)
        macro_avg["Overall"] = defaultdict(float)
        for k, v in macro_avg.items():
            if k != "Overall":
                for metric in v:
                    macro_avg["Overall"][metric] += v[metric]/num_classes

        macro_avgs[key] = macro_avg

    return configs, macro_avgs

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

    for key, val in grid.items():
        grid[key] = list(val)

    return grid

def get_labels(macroavgs):
    """
    Get list of labels. This is needed because some sets doesn't have results for some labels.
    :param macroavgs: A dictionary of macro averages of results that was got from get_configs_and_macroavgs function.
    :return: A list of labels
    """
    labels = set()
    for key, val in macroavgs.items():
        labels.update(val.keys())
    return list(labels)


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
    configurations = list(product(*grid.values()))
    configurations_dicts = [dict(zip(grid.keys(), config)) for config in configurations]

    # Get all the possible lists of pairs(IDs) using inverse_configs dictionary and intersection of them for every configuration.
    pair_list = []
    for dict_config in configurations_dicts:
        list_of_sets = []
        for param_name, val in dict_config.items():
            list_of_sets.append(inverse_configs[param_name][val])

        # Get intersection of sets of IDs for given configurations
        intersection_result = list_of_sets[0]
        # Iterate over the remaining sets and find the intersection
        for s in list_of_sets[1:]:
            intersection_result = intersection_result.intersection(s)

        pair_list.append(list(intersection_result))

    return pair_list

def compare_pairs(list_of_pairs, macroavgs, configs, grid, variable, label_to_show):
    """
    For list of pairs got from get_pairs_to_compare function, compare each pair by plotting bar graphs for given label.
    :param list_of_pairs: got from get_pairs_to_compare function for given variable
    :param macroavgs:
    :param configs:
    :param grid:
    :param variable:
    :param label_to_show: User choice of label (including overall) to show scores in the graph.
    """

    # Form parameter to color dictionary for consistency in color across all pairs
    param_list = list(grid[variable])
    color_list = ['C'+str(i) for i in range(10)]
    param_to_color = {}
    for i in range(len(param_list)):
        param_to_color[str(param_list[i])] = color_list[i]

    # For each pair, form a data dictionary as data = { ID1: [accuracy, precision, recall, f1], ...}
    # and plot a bar graph
    for pair in list_of_pairs:
        scores = macroavgs[pair[0]][label_to_show]
        data = defaultdict(list)
        metric_list = list(scores.keys())
        for id in pair:
            for metric, score in scores.items():
                if label_to_show in macroavgs[id]:
                    data[id].append(macroavgs[id][label_to_show][metric])
                else:
                    data[id].append(0.0)
        data = dict(data)

        # plot a bar graph
        x = np.arange(len(metric_list))  # the label locations
        l = len(data) # length of data (it varies by set)
        width = 1/(l+1)  # the width of the bars
        multiplier = 0

        if l != 0:
            fig, ax = plt.subplots(layout='constrained')
            for id2, scores in data.items():
                id_variable = str(variable) + ": " + str(configs[id2][variable])
                offset = width * multiplier
                rects = ax.bar(x + offset, scores, width, label=id_variable, color=param_to_color[str(configs[id2][variable])])
                ax.bar_label(rects, fmt='%.6s', fontsize='small')
                multiplier += 1

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel('Score')
            ax.set_title(str(label_to_show))
            ax.set_xticks(x + width*(l-1)/2, metric_list)
            ax.legend(loc='upper left', fontsize='small', ncol=l)
            ax.set_ylim(0.0, 1.15)
            # Show information on fixed parameters.
            configs[id2].pop(variable)
            string_configs = ""
            for k, v in configs[id2].items():
                string_configs += str(k) + ": " + str(v) + "\n"
            ax.text(0.99, 0.97, string_configs,
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=ax.transAxes,
                    color='green', fontsize='small')

            plt.show()


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
        "--directory",
        type=str,
        help="Directory with result and configuration files",
        default="",
    )

    args = parser.parse_args()

    # Get necessary dictionaries and lists for processing the comparison.
    configs, macroavgs = get_configs_and_macroavgs(args.directory)
    label_list = get_labels(macroavgs)
    inverse_configs = get_inverse_configs(configs)
    grid = get_grid(configs)

    # Get user inputs and prepare the list of pairs
    choice_variable = user_input_variable(grid)
    choice_label = user_input_label(label_list)
    list_of_pairs = get_pairs_to_compare(grid.copy(), inverse_configs, choice_variable)
    # Show the comparison results of pairs in bar graphs
    compare_pairs(list_of_pairs, macroavgs, configs.copy(), grid, choice_variable, choice_label)