from enum import Enum

class Weights(Enum):
    DINO = "SENTINEL2_ALL_DINO"
    MOCO = "SENTINEL2_ALL_MOCO"
    DECUR = "SENTINEL2_ALL_DECUR"

class Backbones(Enum):
    VIT16 = "vit_small_patch16_224"
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
