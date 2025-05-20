import logging
import torch
import numpy as np
import torch.nn as nn

class InputResizer(nn.Module):
    """
    Resizes input images to the required spatial dimensions using adaptive average pooling.
    """
    def __init__(self, output_size=(224, 224)):
        super().__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size)
        
    def forward(self, x):
        return self.adaptive_pool(x)


def generate_embeddings(model, dataset, mean_time_dim = True, embedding_size = 1024, input_size = 224):
    """
    Iterate over the dataset, run inference on each sample,
    and return a dictionary mapping file names to their embeddings.
    """
    embeddings = {}
    resize = InputResizer(output_size=input_size)
    

    for ind, data_file in enumerate(dataset):

        data = data_file['data'].squeeze(0)
        data = resize(data)

        # average of timestamps
        if mean_time_dim:
            data = data.mean(axis=0)

        if data.ndim == 3:
            # Add a batch dimension if missing
            data = data.unsqueeze(0)

        final_output = model(data)

        if isinstance(final_output, (tuple, list)):
            final_output = final_output[1]  
        else:
            embedding = final_output

        # Flatten and pad the embedding to ensure fixed embedding_size OR average adjacent pairs in embedding
        embedding = embedding.flatten()
        if embedding.ndim == 2:
            embedding = embedding.squeeze()
        if embedding.shape[0] < embedding_size:
            pad_size = embedding_size - embedding.shape[0]
            padding = torch.zeros(pad_size, device=embedding.device, dtype=embedding.dtype)
            embedding = torch.cat([embedding, padding], dim=0)
        if embedding.shape[0] > embedding_size:
            embedding = embedding.view(1024, 2)
            embedding = embedding.mean(dim=1)

        embeddings[data_file['file_name']] = embedding.detach().cpu().numpy().tolist()
        if ind % 100 == 0:
            logging.info("Processed %d files", ind)

    return embeddings
