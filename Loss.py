import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        """
        Initialize the contrastive loss.

        Parameters:
        - temperature (float): Temperature scaling for the logits.
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, q, k, queue):
        """
        Compute the contrastive loss.

        Parameters:
        - q (torch.Tensor): Query embeddings of shape (batch_size, feature_dim).
        - k (torch.Tensor): Positive key embeddings of shape (batch_size, feature_dim).
        - queue (torch.Tensor): Negative key embeddings of shape (queue_size, feature_dim).

        Returns:
        - loss (torch.Tensor): Computed contrastive loss.
        """
        # Positive logits
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        # Negative logits
        l_neg = torch.einsum("nc,ck->nk", [q, queue.clone().detach().T])

        # Combine logits and apply temperature scaling
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature

        # Ground truth labels (positive key is at index 0)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # Cross entropy loss
        loss = F.cross_entropy(logits, labels)
        return loss
    


class MultiHeadContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, alpha=0.5):
        """
        Multi-head contrastive loss for MoCo with neighbor augmentation.

        Parameters:
        - temperature (float): Temperature scaling for the logits.
        - alpha (float): Weight for the central tile loss in the combined loss.
        """
        super(MultiHeadContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight for the central tile vs. neighbors

    def forward(self, q, k_center, k_neighbors, queue, valid_neighbors_count):
        """
        Compute the multi-head contrastive loss.

        Parameters:
        - q (torch.Tensor): Query embeddings of shape [batch_size, feature_dim].
        - k_center (torch.Tensor): Center tile key embeddings of shape [batch_size, feature_dim].
        - k_neighbors (torch.Tensor): Neighbor key embeddings of shape [batch_size, num_neighbors, feature_dim].
        - queue (torch.Tensor): Negative key embeddings of shape [queue_size, feature_dim].
        - valid_neighbors_count (torch.Tensor): Number of valid neighbors for each center tile.

        Returns:
        - loss (torch.Tensor): Combined multi-head contrastive loss.
        """
        batch_size, num_neighbors, feature_dim = k_neighbors.shape

        # Ensure valid_neighbors_count is on the same device as q
        valid_neighbors_count = valid_neighbors_count.to(q.device)

        # Compute positive logits for the center tile
        l_pos_center = torch.einsum("nc,nc->n", [q, k_center]).unsqueeze(-1)  # Shape: [batch_size, 1]

        # Compute positive logits for the neighbors
        l_pos_neighbors = torch.einsum("nc,nkc->nk", [q, k_neighbors])  # Shape: [batch_size, num_neighbors]

        # Average over valid neighbors using valid_neighbors_count
        l_pos_neighbors = l_pos_neighbors.sum(dim=1, keepdim=True) / valid_neighbors_count.unsqueeze(1).clamp(min=1)
        # Shape: [batch_size, 1]

        # Compute negative logits using the queue
        l_neg = torch.einsum("nc,ck->nk", [q, queue.T])  # Shape: [batch_size, queue_size]

        # Combine logits
        logits_center = torch.cat([l_pos_center, l_neg], dim=1)  # [batch_size, 1 + queue_size]
        logits_neighbors = torch.cat([l_pos_neighbors, l_neg], dim=1)  # [batch_size, 1 + queue_size]

        # Apply temperature scaling
        logits_center /= self.temperature
        logits_neighbors /= self.temperature

        # Labels: The positive key is at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=q.device)

        # Compute cross-entropy loss
        loss_center = F.cross_entropy(logits_center, labels)  # Loss for center tile
        loss_neighbors = F.cross_entropy(logits_neighbors, labels)  # Loss for neighbors

        # Combine losses using alpha
        loss = self.alpha * loss_center + (1 - self.alpha) * loss_neighbors
        return loss_center, loss_neighbors, loss