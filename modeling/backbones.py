import sys
from typing import Callable

import torch 
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import resnet50, ResNet50_Weights


class ExtractorModel:
    name: str
    model: torch.nn.Module
    preprocess: Callable


class Vgg16Extractor(ExtractorModel):
    name = "vgg16"
    def __init__(self):
        self.model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.model.classifier = self.model.classifier[:-1]
        self.preprocess = VGG16_Weights.IMAGENET1K_V1.transforms()
        
        
class Resnet50Extractor(ExtractorModel):
    name = "resnet50"
    def __init__(self):
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Identity()
        self.preprocess = ResNet50_Weights.IMAGENET1K_V1.transforms()

model_map = {model.name: model for model in sys.modules[__name__].ExtractorModel.__subclasses__() }
