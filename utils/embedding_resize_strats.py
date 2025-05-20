import torch
import torch.nn as nn

def max_pooling(x):
    """
    Applies max pooling over the patch dimension.
    Args:
        x (torch.Tensor): Raw embeddings of shape (B, num_patches, emb_dim)
    Returns:
        torch.Tensor: Pooled embeddings of shape (B, 1, emb_dim)
    """
    # Transpose to (B, emb_dim, num_patches)
    x = x.transpose(1, 2)
    # Create an adaptive max pooling layer that outputs one value per embedding dimension
    pool = nn.AdaptiveMaxPool1d(1)
    pooled = pool(x)  # shape becomes (B, emb_dim, 1)
    # Transpose back to (B, 1, emb_dim)
    pooled = pooled.transpose(1, 2)
    return pooled

def cnn_max_pooling(x):
    """
    Applies max pooling over the spatial dimensions of a CNN feature map.
    
    Args:
        x (torch.Tensor): Raw convolutional features of shape (B, C, H, W)
    
    Returns:
        torch.Tensor: Pooled embeddings of shape (B, 1, C)
    """
    # Create an adaptive max pooling layer that reduces H and W to 1.
    pool = nn.AdaptiveMaxPool2d((1, 1))
    pooled = pool(x)  # Shape becomes (B, C, 1, 1)
    # Reshape the tensor to (B, 1, C)
    pooled = pooled.view(x.size(0), 1, x.size(1))
    return pooled

def average_adaptive_pooling(x):
    """
    Uses Adaptive Average Pooling over the patch dimension.
    
    Args:
        x (torch.Tensor): Raw embeddings with shape (B, num_patches, emb_dim)
        
    Returns:
        torch.Tensor: Pooled embeddings of shape (B, 1, emb_dim)
    """
    # Transpose to (B, emb_dim, num_patches)
    x_t = x.transpose(1, 2)
    # Adaptive average pool to output 1 value per channel
    pool = nn.AdaptiveAvgPool1d(1)
    pooled = pool(x_t)  # shape: (B, emb_dim, 1)
    # Transpose back to (B, 1, emb_dim)
    return pooled.transpose(1, 2)

def cnn_average_adaptive_pooling(x):
    """
    Uses AdaptiveAvgPool1d over the patch dimension.

    Args:
        x (torch.Tensor): Raw embeddings with shape (B, num_patches, emb_dim)

    Returns:
        torch.Tensor: Pooled embeddings of shape (B, 1, emb_dim)
    """
    pool = nn.AdaptiveAvgPool2d((1, 1))
    out  = pool(x)                   # → (B, C, 1, 1)
    return out.view(x.size(0), 1, x.size(1))

def mean_pooling(x):
    """
    Performs mean pooling over the patch dimension.
    
    Args:
        x (torch.Tensor): Raw embeddings with shape (B, num_patches, emb_dim)
        
    Returns:
        torch.Tensor: Pooled embeddings of shape (B, 1, emb_dim)
    """
    # Simply take the mean over the patch (dim=1)
    return torch.mean(x, dim=1, keepdim=True)

def cnn_mean_pooling(x):
    return torch.mean(x, dim=[2, 3])

def min_pooling(x):
    """
    Performs min pooling over the patch dimension.
    
    Args:
        x (torch.Tensor): Raw embeddings with shape (B, num_patches, emb_dim)
        
    Returns:
        torch.Tensor: Pooled embeddings of shape (B, 1, emb_dim)
    """
    # Compute the minimum along the patch dimension.
    pooled, _ = torch.min(x, dim=1, keepdim=True)
    return pooled

def cnn_min_pooling(x):
    return torch.amin(x, dim=(2, 3))

class AttentionPooling(nn.Module):
    """
    An example attention pooling module. It computes attention scores for each patch embedding,
    then uses a weighted sum to produce a pooled representation.
    """
    def __init__(self, emb_dim):
        """
        Args:
            emb_dim (int): The embedding dimension for each patch.
        """
        super().__init__()
        # Project each embedding to a scalar score.
        self.attn = nn.Linear(emb_dim, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Raw embeddings with shape (B, num_patches, emb_dim)
        
        Returns:
            torch.Tensor: Pooled embedding of shape (B, 1, emb_dim)
        """
        # Compute raw attention scores for each patch.
        scores = self.attn(x)  # shape: (B, num_patches, 1)
        # Normalize scores to probabilities over patches.
        weights = torch.softmax(scores, dim=1)  # shape: (B, num_patches, 1)
        # Weighted sum of the patch embeddings.
        pooled = torch.sum(x * weights, dim=1, keepdim=True)  # shape: (B, 1, emb_dim)
        return pooled

# Usage of attention pooling:
    # # Attention pooling: create an instance of the AttentionPooling module
    # attn_pool_module = AttentionPooling(emb_dim)
    # pooled_attn = attn_pool_module(raw_embeddings)
    # print("Attention Pooling:", pooled_attn.shape)  # (8, 1, 768)