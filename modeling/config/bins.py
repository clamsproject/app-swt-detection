from typing import List, Dict

# Binning behavior: Labels not explicitly included in any bin map to the NEGATIVE class ("-").
# Empty binning ({}) triggers full granular classification (18 classes + negative).
# Implementation: see pretraining_bin() in modeling/train.py

nobinning = {}

label_bins = {
    "Bars": ["B"],
    "Chyron-other": ["Y", "KU"],  #"U", "K"],
    "Chyron-person": ["IN"],  # ["I", "N"],
    "Credits": ["CR"],  # ["C", "R"],
    "Main": ["M"],
    "Opening": ["O", "W"],
    "Slate": ["S"],  # "S:H", "S:C", "S:D", "S:B", "S:G"],
    "Other-text-sm": ["L", "G", "F", "E", "T"],
    "Other-text-md": ["M", "F", "E", "GLOTW"],  #"W", "T", "O", "L", "G"],
    "Neg": ["P", "-"],
}
label_bins["Other-text-lg"] = label_bins["Chyron-other"] + label_bins["Other-text-md"]

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
        "-": label_bins["Neg"],
    },

    "simpler": {
        "Bars": label_bins["Bars"],
        "Slate": label_bins["Slate"],
        "Chyron": label_bins["Chyron-person"],
        "Credits": label_bins["Credits"],
        "Neg": label_bins["Neg"],
    },
    
    "simple": {
        "Bars": label_bins["Bars"],
        "Slate": label_bins["Slate"],
        "Chyron-person": label_bins["Chyron-person"],
        "Credits": label_bins["Credits"],
        "Other-text": label_bins["Other-text-lg"],
        "Neg": label_bins["Neg"],
    },

    "relaxed": {
        "Bars": label_bins["Bars"],
        "Slate": label_bins["Slate"],
        "Chyron": label_bins["Chyron-other"] + label_bins["Chyron-person"],
        "Credits": label_bins["Credits"],
        "Other-text": label_bins["Other-text-md"],
        "Neg": label_bins["Neg"],
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

    # 1) "Collapse close categories"
    # Idea:  Collapse categories like G, L, O, T, that are ambiguous to
    # humans, and group categories that depend on the content of the text.
    "collapse-close": {
        "GLOTW": ["G", "L", "O", "T", "W"],
        "CR": ["C", "R"],
        "IN": ["I", "N"],
        "KU": ["K", "U"],
        "B": ["B"],
        "S": ["S"],
        "M": ["M"],
        "Y": ["Y"],
        "F": ["F"],
        "E": ["E"],
        "P": ["P"]
    },

    # WARN: the schemes below were experimental during v8.0 development and 
    # no longer in use, kept here for historical reasons.
    # unlike "collapse-close", at least one of the "collapsed" bin names in
    # the following schemes contains multiple top-leve "label-bins" categories.
    # hence they are not suitable for "post bin" 
    
    # e.g., INKU in "collapse-close-reduce-difficulty" contains both
    # "Chyron-person" and "Chyron-other" categories
    
    # to handle this problem, for now I will just disable these schemes from
    # being used as "post bin" schemes in the `tfLabelMapPreset` parameter.
    
    # 2) "Collapse close categories and reduce difficulty"
    # Idea: Reduce difficulty by grouping some lower thirds
    "collapse-close-reduce-difficulty": {
        "GLOTW": ["G", "L", "O", "T", "W"],
        "CR": ["C", "R"],
        "IKNU": ["I", "K", "N", "U"],
        "B": ["B"],
        "S": ["S"],
        "M": ["M"],
        "Y": ["Y"],
        "F": ["F"],
        "E": ["E"],
        "P": ["P"]
    },

    # 3) "Collapse close categories and group all lower thirds"
    # Idea: Further reduce difficulty by grouping all lower thirds
    "collapse-close-bin-lower-thirds": {
        "GLOTW": ["G", "L", "O", "T", "W"],
        "CR": ["C", "R"],
        "IKNUY": ["I", "K", "N", "U", "Y"],
        "B": ["B"],
        "S": ["S"],
        "M": ["M"],
        "F": ["F"],
        "E": ["E"],
        "P": ["P"]
    },

    # 4) "Ignore difficult distinctions"
    # Idea: Any more binning would reduce the value of categories too much
    "ignore-difficulties": {
        "GLMOTW": ["G", "L", "M", "O", "T", "W"],
        "CR": ["C", "R"],
        "IKNUY": ["I", "K", "N", "U", "Y"],
        "B": ["B"],
        "S": ["S"],
        "F": ["F"],
        "E": ["E"],
        "P": ["P"]
    }

}
