from typing import List, Dict

# Binning behavior: Labels not explicitly included in any bin map to the NEGATIVE class ("-").
# Empty binning ({}) triggers full granular classification (18 classes + negative).
# Implementation: see pretraining_bin() in modeling/train.py

nobinning = {}

label_bins = {
    "Bars": ["B"],
    "Chyron-other": ["Y", "U", "K"],
    "Chyron-person": ["I", "N"],
    "Credits": ["C", "R"],
    "Main": ["M"],
    "Opening": ["O", "W"],
    "Slate": ["S", "S:H", "S:C", "S:D", "S:B", "S:G"],
    "Other-text-sm": ["L", "G", "F", "E", "T"],
    "Other-text-md": ["M", "O", "W", "L", "G", "F", "E", "T"],
    "Other-text-lg": ["M", "O", "W", "Y", "U", "K", "L", "G", "F", "E", "T"],
}

binning_schemes: Dict[str, Dict[str, List[str]]] = {
    "noprebin": nobinning,
    "nomap": nobinning,

    "strict": {
        "Bars": label_bins["Bars"],
        "Slate": label_bins["Slate"],
        "Chyron-person": label_bins["Chyron-person"],
        "Credits": label_bins["Credits"],
        "Main": label_bins["Main"],
        "Opening": label_bins["Opening"],
        "Chyron-other": label_bins["Chyron-other"],
        "Other-text": label_bins["Other-text-sm"],
    },

    "simpler": {
        "Bars": label_bins["Bars"],
        "Slate": label_bins["Slate"],
        "Chyron": label_bins["Chyron-person"],
        "Credits": label_bins["Credits"],
    },
    
    "simple": {
        "Bars": label_bins["Bars"],
        "Slate": label_bins["Slate"],
        "Chyron-person": label_bins["Chyron-person"],
        "Credits": label_bins["Credits"],
        "Other-text": label_bins["Other-text-lg"],
    },

    "relaxed": {
        "Bars": label_bins["Bars"],
        "Slate": label_bins["Slate"],
        "Chyron": label_bins["Chyron-other"] + label_bins["Chyron-person"],
        "Credits": label_bins["Credits"],
        "Other-text": label_bins["Other-text-md"],
    },

    "binary-bars": {
        "Bars": label_bins["Bars"],
    },

    "binary-slate": {
        "Slate": label_bins["Slate"],
    },

    "binary-chyron-strict": {
        "Chyron-person": label_bins["Chyron-person"],
    },

    "binary-chyron-relaxed": {
        "Chyron": label_bins["Chyron-other"] + label_bins["Chyron-person"],
    },

    "binary-credits": {
        "Credits": label_bins["Credits"],
    },

    # Human ambiguity-based binning schemes (summer 2025)

    # 1) "Collapse human ambiguity"
    # Idea: This binning comes from human observations about category ambiguity.
    # These pre-bin categories are thought to be difficult/ambiguous for humans.
    # This binning is not at all informed by training eval numbers, but based on
    # observations about inter-labeler consistency during labeling and answering
    # interns questions about labeling. This binning preserves most of the
    # distinctions about which humans are reliable judges, and it preserves the
    # most high value categorizations.
    # Bin map: "GLOTW": ["G", "L", "O", "T", "W"]
    "collapse-human-ambiguity": {
        "GLOTW": ["G", "L", "O", "T", "W"],
        "B": ["B"],
        "S": ["S"],
        "I": ["I"],
        "C": ["C"],
        "R": ["R"],
        "M": ["M"],
        "N": ["N"],
        "Y": ["Y"],
        "U": ["U"],
        "K": ["K"],
        "F": ["F"],
        "E": ["E"],
        "P": ["P"]
    },

    # 2) "Collapse human ambiguity and reduce difficulty"
    # Idea: Further bin by combining a couple of the most challenging categories
    # along with the ones they seem to visually resemble.
    # Bin map: "GLOTW": ["G", "L", "O", "T", "W"], "CR": ["C", "R"], "IN": ["I", "N"]
    "collapse-ambiguity-reduce-difficulty": {
        "GLOTW": ["G", "L", "O", "T", "W"],
        "CR": ["C", "R"],
        "IN": ["I", "N"],
        "B": ["B"],
        "S": ["S"],
        "M": ["M"],
        "Y": ["Y"],
        "U": ["U"],
        "K": ["K"],
        "F": ["F"],
        "E": ["E"],
        "P": ["P"]
    },

    # 3) "Collapse human ambiguity and further reduce difficulty"
    # Idea: Just like 2, except that it also expects that K and U are difficult
    # categories and speculates that they are best collapsed with E.
    # Bin map: "GLOTW": ["G", "L", "O", "T", "W"], "EKU": ["E", "K", "U"],
    #          "CR": ["C", "R"], "IN": ["I", "N"]
    "collapse-ambiguity-further-reduce": {
        "GLOTW": ["G", "L", "O", "T", "W"],
        "EKU": ["E", "K", "U"],
        "CR": ["C", "R"],
        "IN": ["I", "N"],
        "B": ["B"],
        "S": ["S"],
        "M": ["M"],
        "Y": ["Y"],
        "F": ["F"],
        "P": ["P"]
    },

    # 4) "Ignore all difficult distinctions"
    # Idea: Further bin by combining most of the tricky categories into a large
    # catch-all SWT category.
    # Bin map: "EGLMOTWY": ["E", "G", "L", "K", "M", "O", "T", "U", "W", "Y"],
    #          "CR": ["C", "R"], "IN": ["I", "N"]
    "ignore-difficult-distinctions": {
        "EGLMOTWY": ["E", "G", "L", "K", "M", "O", "T", "U", "W", "Y"],
        "CR": ["C", "R"],
        "IN": ["I", "N"],
        "B": ["B"],
        "S": ["S"],
        "P": ["P"],
        "F": ["F"]
    }
}
