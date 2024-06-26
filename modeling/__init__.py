negative_label = '-'
positive_label = '+'

# full typology from https://github.com/clamsproject/app-swt-detection/issues/1
FRAME_TYPES = ["B", "S", "W", "L", "O",
               "M", "I", "N", "E", "P", "Y", "K", "G", "T", "F", "C", "R"]
FRAME_TYPES_WITH_SUBTYPES = ["B", "SH", "SC", "SD", "SB", "SG", "W", "L", "O",
                             "M", "I", "N", "E", "P", "Y", "K", "G", "T", "F", "C", "R"]

# These are time frames that are typically static (that is, the text does not
# move around or change as with rolling credits). These are frame names after
# the label mapping.
static_frames = ['bars', 'slate', 'chyron']
