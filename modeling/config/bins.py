from typing import List, Dict

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
    }
}
