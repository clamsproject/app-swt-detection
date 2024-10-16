negative_label = '-'
positive_label = '+'

# full typology from https://github.com/clamsproject/app-swt-detection/issues/1
FRAME_TYPES = [
    "B", "S", "I", "C", "R", "M", "O", "W",
    "N", "Y", "U", "K",
    "L", "G", "F", "E", "T",
    "P",
]
FRAME_TYPES_WITH_SUBTYPES = FRAME_TYPES.copy() + ['SH', 'SC', 'SD', 'SB', 'SG']
FRAME_TYPES_WITH_SUBTYPES.remove('S')

# These are time frames that are typically static (that is, the text does not
# move around or change as with rolling credits). These are frame names after
# the label mapping.
static_frames = ['bars', 'slate', 'chyron']
