from enum import Enum

class Weights(Enum):
    DINO = "SENTINEL2_ALL_DINO"
    MOCO = "SENTINEL2_ALL_MOCO"
    DECUR = "SENTINEL2_ALL_DECUR"
    FGMAE = "SENTINEL2_ALL_FGMAE"
    MAE = "SENTINEL2_ALL_MAE"
    SOFTCON= "SENTINEL2_ALL_SOFTCON"
    

class Backbones(Enum):
    VIT16 = "vit_small_patch16_224"
    VITBASE = "vit_base_patch16_224"
    VITLARGE = "vit_large_patch16_224"
    VITHUGE = "vit_huge_patch14_224"
    VIT14SMALL = "vit_small_patch14_dinov2"
    VIT14BASE = "vit_base_patch14_dinov2"
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
