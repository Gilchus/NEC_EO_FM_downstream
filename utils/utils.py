import pandas as pd
import torch.nn as nn
import torch
import timm
import logging
from enums import Weights, Backbones
from torchgeo.models import ViTSmall16_Weights, ResNet18_Weights, ResNet50_Weights


def create_submission_from_dict(emb_dict):
    """Assume dictionary has format {hash-id0: embedding0, hash-id1: embedding1, ...}
    """
    df_submission = pd.DataFrame.from_dict(emb_dict, orient='index')
    
    # Reset index with name 'id'
    df_submission.index.name = 'id'
    df_submission.reset_index(drop=False, inplace=True)
        
    return df_submission

def test_submission(path_to_submission: str, 
                        expected_embedding_ids: set, 
                        embedding_dim: int = 1024):
        # Load data
        df = pd.read_csv(path_to_submission, header=0)

        # Verify that id is in columns
        if 'id' not in df.columns:
            raise ValueError(f"""Submission file must contain column 'id'.""")

        # Temporarily set index to 'id'
        df.set_index('id', inplace=True)

        # Check that all samples are included
        submitted_embeddings = set(df.index.to_list())
        n_missing_embeddings = len(expected_embedding_ids.difference(submitted_embeddings))
        if n_missing_embeddings > 0:
            raise ValueError(f"""Submission is missing {n_missing_embeddings} embeddings.""")
        
        # Check that embeddings have the correct length
        if len(df.columns) != embedding_dim:
            raise ValueError(f"""{embedding_dim} embedding dimensions, but provided embeddings have {len(df.columns)} dimensions.""")

        # Convert columns to float
        try:
            for col in df.columns:
                df[col] = df[col].astype(float)
        except Exception as e:
            raise ValueError(f"""Failed to convert embedding values to float.
        Check embeddings for any not-allowed character, for example empty strings, letters, etc.
        Original error message: {e}""")

        # Check if any NaNs 
        if df.isna().any().any():
            raise ValueError(f"""Embeddings contain NaN values.""")
        
        # Successful completion of the function
        return True

# def load_model(backbone: Backbones, weights: Weights):
#     if backbone == Backbones.VIT16:
#         weight_class = ViTSmall16_Weights
#     elif backbone == Backbones.RESNET18:
#         weight_class = ResNet18_Weights
#     elif backbone == Backbones.RESNET50:
#         weight_class = ResNet50_Weights
#     else:
#         raise ValueError(f"Backbone {backbone.name} is not supported.")

#     weights_obj = getattr(weight_class, weights.value)

#     model = timm.create_model(backbone.value, in_chans=weights_obj.meta["in_chans"])

#     state_dict = weights_obj.get_state_dict(progress=True)
#     model.load_state_dict(state_dict, strict=False)
#     model.eval()

#     logging.info("Loaded TorchGeo model.")
#     return model

def load_model(backbone: Backbones, weights: Weights,
               avg_fn: callable = None,
               concat_avg: bool = False):

    if backbone == Backbones.VIT16:
        weight_class = ViTSmall16_Weights
    elif backbone == Backbones.RESNET18:
        weight_class = ResNet18_Weights
    elif backbone == Backbones.RESNET50:
        weight_class = ResNet50_Weights
    else:
        raise ValueError(f"Backbone {backbone.name} is not supported.")

    weights_obj = getattr(weight_class, weights.value)
    model = timm.create_model(backbone.value, in_chans=weights_obj.meta["in_chans"])
    state_dict = weights_obj.get_state_dict(progress=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    logging.info("Loaded TorchGeo model.")

    if backbone == Backbones.VIT16:
        # Custom forwarding for ViT models
        if hasattr(model, "get_intermediate_layers"):
            orig_get_intermediate_layers = model.get_intermediate_layers

            def new_forward(x):
                # Get the output from the last transformer block (using 1 layer by default)
                with torch.no_grad():
                    intermediate_outputs = orig_get_intermediate_layers(x, 1)
                last_output = intermediate_outputs[-1]  # shape: (B, tokens, emb_dim)
                # Exclude the [CLS] token
                raw_embeddings = last_output[:, 1:, :]  # shape: (B, num_patches, emb_dim)
                logging.info(f"Raw embedding shape:{raw_embeddings.shape}")

                if avg_fn is not None:
                    pooled = avg_fn(raw_embeddings)
                else:
                    pooled = torch.mean(raw_embeddings, dim=1, keepdim=True)  # shape: (B, 1, emb_dim)
                if concat_avg:
                    # Concatenate the [CLS] token (first token of last_output) with the pooled result.
                    cls_token = last_output[:, :1, :]  # shape: (B, 1, emb_dim)
                    output = torch.cat([cls_token, pooled], dim=-1)  # shape: (B, 1, 2*emb_dim)
                    logging.info(f"Pooled embedding shape with CLS: {output.shape}")
                else:
                    output = pooled
                    logging.info(f"Pooled embedding shape: {output.shape}")

                return output

        model.forward = new_forward
        logging.info(f"Modified forward to return raw patch embeddings with custom averaging.")

    elif backbone in [Backbones.RESNET18, Backbones.RESNET50]:
        # Many timm ResNet models implement a forward_features() method that returns the feature maps 
        # (i.e. the output from the last conv layer, before global pooling and fc)
        def new_forward(x):
            # Get the raw convolutional feature maps
            feats = model.forward_features(x)  # typically shape [B, C, H, W]
            logging.info(f"Last conv features shape: {feats.shape}")
            # Apply the custom pooling if provided, else perform global average pooling.
            if avg_fn is not None:
                pooled = avg_fn(feats)
            else:
                pooled = feats.mean(dim=[2, 3])  # Global average pooling over spatial dims.
            logging.info(f"Pooled features shape: {pooled.shape}")
            return pooled

        model.forward = new_forward
        logging.info("Modified forward for ResNet to return last convolutional layer output.")
    else:
        logging.warning("Model does not support custom intermediate output extraction; forward method not modified.")


    return model