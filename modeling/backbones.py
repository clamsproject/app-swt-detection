# ==============================================================================|
# Imports
import sys
from typing import Callable

import torch
# ConvNext Models
from torchvision.models import convnext_base, ConvNeXt_Base_Weights  # ConvNeXt BASE
from torchvision.models import convnext_large, ConvNeXt_Large_Weights  # ConvNeXt LARGE
from torchvision.models import convnext_small, ConvNeXt_Small_Weights  # ConvNeXt SMALL
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights  # ConvNeXt TINY
# DenseNet Models
from torchvision.models import densenet121, DenseNet121_Weights  # DenseNet 121
from torchvision.models import densenet161, DenseNet161_Weights  # DenseNet 161
from torchvision.models import densenet169, DenseNet169_Weights  # DenseNet 169
from torchvision.models import densenet201, DenseNet201_Weights  # DenseNet 201
# EfficientNet Models
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights  # EfficientNetV2 L
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights  # EfficientNetV2 M
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights  # EfficientNetV2 S
# Inceptionv3
from torchvision.models import inception_v3, Inception_V3_Weights  # Inception v3
from torchvision.models import resnet101, ResNet101_Weights  # Resnet 101
from torchvision.models import resnet152, ResNet152_Weights  # Resnet 152
# ResNet Models
from torchvision.models import resnet18, ResNet18_Weights  # Resnet 18
from torchvision.models import resnet50, ResNet50_Weights  # Resnet 50
# VGG Models
from torchvision.models import vgg16, VGG16_Weights  # VGG16
from torchvision.models import vgg16_bn, VGG16_BN_Weights  # VGG16 (Batch Normalized)
from torchvision.models import vgg19, VGG19_Weights  # VGG19
from torchvision.models import vgg19_bn, VGG19_BN_Weights  # VGG19 (Batch Normalized)


# ===========================================================================|
# Base Class
class ExtractorModel:
    name: str
    model: torch.nn.Module
    preprocess: Callable


# ===========================================================================|
# Models
# TODO/REVIEW - do we want to be able to change the weight versions (IMAGENET1K_V1 etc)
# ==========================================|
# ConvNext Models
class ConvnextBaseExtractor(ExtractorModel):
    name = "convnext_base"

    def __init__(self):
        self.model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.model.classifier[-1] = torch.nn.Identity()
        self.preprocess = ConvNeXt_Base_Weights.IMAGENET1K_V1.transforms()


class ConvnextTinyExtractor(ExtractorModel):
    name = "convnext_tiny"

    def __init__(self):
        self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.model.classifier[-1] = torch.nn.Identity()
        self.preprocess = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()


class ConvnextSmallExtractor(ExtractorModel):
    name = "convnext_small"

    def __init__(self):
        self.model = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
        self.model.classifier[-1] = torch.nn.Identity()
        self.preprocess = ConvNeXt_Small_Weights.IMAGENET1K_V1.transforms()


class ConvnextLargeExtractor(ExtractorModel):
    name = "convnext_lg"

    def __init__(self):
        self.model = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)
        self.model.classifier[-1] = torch.nn.Identity()
        self.preprocess = ConvNeXt_Large_Weights.IMAGENET1K_V1.transforms()


# ==========================================|
# DenseNet Models
class Densenet121Extractor(ExtractorModel):
    name = "densenet121"

    def __init__(self):
        self.model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.model.classifier = torch.nn.Identity()
        self.preprocess = DenseNet121_Weights.IMAGENET1K_V1.transforms()


class Densenet161Extractor():
    name = "densenet161"

    def __init__(self):
        self.model = densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1)
        self.model.classifier = torch.nn.Identity()
        self.preprocess = DenseNet161_Weights.IMAGENET1K_V1.transforms()


class Densenet169Extractor():
    name = "densenet169"

    def __init__(self):
        self.model = densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)
        self.model.classifier = torch.nn.Identity()
        self.preprocess = DenseNet169_Weights.IMAGENET1K_V1.transforms()


class Densenet201Extractor():
    name = "densenet201"

    def __init__(self):
        self.model = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
        self.model.classifier = torch.nn.Identity()
        self.preprocess = DenseNet201_Weights.IMAGENET1K_V1.transforms()


# ==========================================|
# EfficientNet Models
class EfficientnetSmallExtractor(ExtractorModel):
    name = "efficientnet_small"

    def __init__(self):
        self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.model.classifier = torch.nn.Identity()
        self.preprocess = EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()


class EfficientnetMediumExtractor(ExtractorModel):
    name = "efficientnet_med"

    def __init__(self):
        self.model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        self.model.classifier = torch.nn.Identity()
        self.preprocess = EfficientNet_V2_M_Weights.IMAGENET1K_V1.transforms()


class EfficientnetLargeExtractor(ExtractorModel):
    name = "efficientnet_large"

    def __init__(self):
        self.model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        self.model.classifier = torch.nn.Identity()
        self.preprocess = EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms()


# ==========================================|
# Inception Model
class InceptionV3Extractor(ExtractorModel):
    name = "inceptionv3"

    def __init__(self):
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Identity()
        self.preprocess = Inception_V3_Weights.IMAGENET1K_V1.transforms()


# ==========================================|
# Resnet Models

class Resnet18Extractor(ExtractorModel):
    name = "resnet18"

    def __init__(self):
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Identity()
        self.preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()


class Resnet50Extractor(ExtractorModel):
    name = "resnet50"

    def __init__(self):
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Identity()
        self.preprocess = ResNet50_Weights.IMAGENET1K_V1.transforms()


class Resnet101Extractor(ExtractorModel):
    name = "resnet101"

    def __init__(self):
        self.model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Identity()
        self.preprocess = ResNet101_Weights.IMAGENET1K_V1.transforms()


class Resnet152Extractor(ExtractorModel):
    name = "resnet152"

    def __init__(self):
        self.model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Identity()
        self.preprocess = ResNet152_Weights.IMAGENET1K_V1.transforms()


# ==========================================|
# VGG Models
class Vgg16Extractor(ExtractorModel):
    name = "vgg16"

    def __init__(self):
        self.model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.model.classifier = self.model.classifier[:-1]
        self.preprocess = VGG16_Weights.IMAGENET1K_V1.transforms()


class BN_Vgg16Extractor(ExtractorModel):
    name = "bn_vgg16"

    def __init__(self):
        self.model = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        self.model.classifier = self.model.classifier[:-1]
        self.preprocess = VGG16_BN_Weights.IMAGENET1K_V1.transforms()


class Vgg19Extractor(ExtractorModel):
    name = "vgg19"

    def __init__(self):
        self.model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.model.classifier = self.model.classifier[:-1]
        self.preprocess = VGG19_Weights.IMAGENET1K_V1.transforms()


class BN_VGG19Extractor(ExtractorModel):
    name = "bn_vgg19"

    def __init__(self):
        self.model = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)
        self.model.classifier = self.model.classifier[:-1]
        self.preprocess = VGG19_BN_Weights.IMAGENET1K_V1.transforms()


# ===========================================================================|

model_map = {model.name: model for model in sys.modules[__name__].ExtractorModel.__subclasses__()}
