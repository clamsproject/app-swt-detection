# ==============================================================================|
# Imports
import sys
from typing import Callable

from transformers import AutoImageProcessor, ConvNextV2Model, ConvNextModel


# ===========================================================================|
# Base Class
class ExtractorModel:
    name: str
    dim: int
    model: Callable
    preprocess: Callable


# ==========================================|
# ConvNeXt Models
# Using pretrained/unsupervised ones from huggingface (facebook official) models
## all these backbones are expecting 224x224 input images
# size variance for v1; T S B L XL
# size variances for v2; A F P N T B L H 

# This class does NOT inherit from ExtractorModel, so it won't be picked up
# by code in the main block iterating through ExtractorModel.__subclasses__().
class _ConvnextExtractorInitBase:
    def _init_hf_convnext_components(self):
        model_version = 2 if 'v2' in self.name else 1

        if model_version == 2:
            full_model_name = f"facebook/{self.name.replace('_', '-')}-22k-224"
            self.model = ConvNextV2Model.from_pretrained(full_model_name)
        else:
            full_model_name = f"facebook/{self.name.replace('_', '-')}-224"
            self.model = ConvNextModel.from_pretrained(full_model_name)
        # self.preprocessor is stored as an instance variable, though only used within the lambda.
        # This is fine, or the lambda could capture AutoImageProcessor.from_pretrained directly if preferred.
        self.preprocessor = AutoImageProcessor.from_pretrained(full_model_name)
        self.preprocess = lambda image_input: self.preprocessor(image_input, return_tensors="pt")["pixel_values"]  # will return [num_images, 3, 224, 224] shaped torsors


class ConvnextTinyExtractor(ExtractorModel, _ConvnextExtractorInitBase):
    name = "convnext_tiny"
    dim = 768

    def __init__(self):
        self._init_hf_convnext_components()


class ConvnextSmallExtractor(ExtractorModel, _ConvnextExtractorInitBase):
    name = "convnext_base"
    dim = 1024

    def __init__(self):
        self._init_hf_convnext_components()


class ConvnextLargeExtractor(ExtractorModel, _ConvnextExtractorInitBase):
    name = "convnext_large"
    dim = 1536

    def __init__(self):
        self._init_hf_convnext_components()


class ConvnextV2TinyExtractor(ExtractorModel, _ConvnextExtractorInitBase):
    name = "convnextv2_tiny"
    dim = 768  # Output dimension for ConvNextV2-Tiny

    def __init__(self):
        self._init_hf_convnext_components()


class ConvnextV2SmallExtractor(ExtractorModel, _ConvnextExtractorInitBase):
    name = "convnextv2_base"
    dim = 1024

    def __init__(self):
        self._init_hf_convnext_components()


class ConvnextV2LargeExtractor(ExtractorModel, _ConvnextExtractorInitBase):
    name = "convnextv2_large"
    dim = 1536

    def __init__(self):
        self._init_hf_convnext_components()

model_map = {
    model.name: model for model
    in sys.modules[__name__].ExtractorModel.__subclasses__() if model.name != 'inceptionv3'}

model_dim_map = {
    model.name: model.dim for model
    in sys.modules[__name__].ExtractorModel.__subclasses__() if model.name != 'inceptionv3'}

if __name__ == "__main__":
    for name in model_map.keys():
        print(name)
        