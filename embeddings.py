import os
import logging
import argparse
import pytorch_lightning as pl
from torchvision import transforms
import timm
import torch
from torchgeo.models import ViTSmall16_Weights, ResNet18_Weights, ResNet50_Weights
from data import E2SChallengeDataset, S2L2A_MEAN, S2L2A_STD, S2L1C_MEAN, S2L1C_STD, S1GRD_MEAN, S1GRD_STD
from utils import create_submission_from_dict, test_submission, load_model
from enums import Weights, Backbones
from get_embeddings import generate_embeddings
from embedding_resize_strats import max_pooling, min_pooling, mean_pooling, average_adaptive_pooling, cnn_max_pooling, cnn_mean_pooling, cnn_min_pooling, cnn_average_adaptive_pooling


#TEST DATA PATH
#DATA_PATH = "/dccstor/geofm-finetuning/luisgilch/ssl4eo_challenge/test_image"

DATA_PATH = "/dccstor/geofm-datasets/datasets/Embed2Scale_Challenge/SSL4EO-S12-downstream/data_eval"
OUTPUT_PATH = "/dccstor/geofm-finetuning/luisgilch/ssl4eo_challenge/outputs"

class Normalize:
    def __call__(self, img):
        img = img.float()
        img_norm = (img / 10000) 
        img_norm = torch.clamp(img_norm, 0, 1)
        return img_norm


resize_fn_map = {
    "mean_pooling": mean_pooling,
    "max_pooling": max_pooling,
    "min_pooling": min_pooling,
    "average_adaptive_pooling": average_adaptive_pooling,
}

cnn_resize_fn_map = {
    "mean_pooling": cnn_mean_pooling,
    "max_pooling": cnn_max_pooling,
    "min_pooling": cnn_min_pooling,
    "average_adaptive_pooling": cnn_average_adaptive_pooling,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load backbone and weights for TorchGeo."
    )
    parser.add_argument("--backbone", type=str,
                        choices=[b.name for b in Backbones],
                        required=True,
                        help="Backbone to use: " + ", ".join([b.name for b in Backbones]))
    parser.add_argument("--weights", type=str,
                        choices=[w.name for w in Weights],
                        required=True,
                        help="Weights to use: " + ", ".join([w.name for w in Weights]))
    parser.add_argument("--resize", type=str,
                        choices=resize_fn_map.keys(),
                        required=True,
                        help="Resize strategy to use: " + ", ".join(resize_fn_map.keys()))
    parser.add_argument("--concatcls", action="store_true",
                        help="Concat CLS token for applicable models (default: False)")
    args = parser.parse_args()

    if Backbones[args.backbone] in [Backbones.RESNET18, Backbones.RESNET50]:
        resize_fn = cnn_resize_fn_map[args.resize]
    else:
        resize_fn = resize_fn_map[args.resize]

    return Backbones[args.backbone], Weights[args.weights], resize_fn, args.concatcls
        
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    
    pl.seed_everything(42, workers=True)
    logger.info("Random seeds set.")

    backbone, weights, resize_strat, concat_cls = parse_args()
    encoder = load_model(backbone=backbone, weights=weights, avg_fn=resize_strat, concat_avg=concat_cls)
    logger.info(f"Backbone: {backbone}, Weights: {weights}, ResizeStrat: {resize_strat}, UseCLS: {concat_cls}")

    # Prepare the dataset and transformation
    data_transform = transforms.Compose([
        Normalize()
    ])

    dataset = E2SChallengeDataset(
        DATA_PATH,
        modalities=["s2l1c"],
        seasons = 4,
        dataset_name='bands',
        transform=data_transform,
        concat=True,
        output_file_name=True
    )
    logger.info("Length of dataset: %d", len(dataset))
    logger.info("Sample shape: %s", dataset[0]['data'].shape)

    # Generate embeddings using the provided input size
    embeddings = generate_embeddings(model = encoder, dataset = dataset, mean_time_dim = True, input_size = 224)

    submission_file = create_submission_from_dict(embeddings)
    logger.info("Number of embeddings: %d", len(submission_file))

    resize_name = next(k for k, v in cnn_resize_fn_map.items() if v == resize_strat)
    if concat_cls:
        fn = f"{backbone.name}_{weights.name}_{resize_name}_cls_embeddings.csv"
    else:
        fn = f"{backbone.name}_{weights.name}_{resize_name}_embeddings.csv"
    fp = os.path.join(OUTPUT_PATH, fn)

    submission_file.to_csv(fp, index=False)
    logger.info("Embeddings saved to %s", OUTPUT_PATH)

    # Validate the submission 
    embedding_ids = set(embeddings.keys())
    assert test_submission(fp, embedding_ids, 1024)

if __name__ == "__main__":
    main()