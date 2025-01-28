import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseContrastiveLoss(nn.Module):
    def __init__(self):
        """
        Base class for contrastive loss functions.
        """
        super(BaseContrastiveLoss, self).__init__()

    def loss_orginal(self, q, k, queue):
        """
        Compute the original contrastive loss.

        Parameters:
        - q (torch.Tensor): Query embeddings.
        - k (torch.Tensor): Positive key embeddings.
        - queue (torch.Tensor): Negative key embeddings.

        Returns:
        - loss (torch.Tensor): Contrastive loss.
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


class ContrastiveLoss(BaseContrastiveLoss):
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
        - q (torch.Tensor): Query embeddings.
        - k (torch.Tensor): Positive key embeddings.
        - queue (torch.Tensor): Negative key embeddings.

        Returns:
        - loss (torch.Tensor): Computed contrastive loss.
        """
        return self.loss_orginal(q, k, queue)


class MultiHeadContrastiveLoss(BaseContrastiveLoss):
    def __init__(self, temperature=0.07, alpha=0.5):
        """
        Multi-head contrastive loss for MoCo with neighbor augmentation.

        Parameters:
        - temperature (float): Temperature scaling for the logits.
        - alpha (float): Weight for the central tile loss in the combined loss.
        """
        super(MultiHeadContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def loss_neighbor(self, q, k_neighbors, queue, valid_neighbors_count):
        """
        Compute the neighbor contrastive loss.

        Parameters:
        - q (torch.Tensor): Query embeddings.
        - k_neighbors (torch.Tensor): Neighbor key embeddings.
        - queue (torch.Tensor): Negative key embeddings.
        - valid_neighbors_count (torch.Tensor): Number of valid neighbors.

        Returns:
        - loss (torch.Tensor): Neighbor contrastive loss.
        """
        # Positive logits for neighbors
        l_pos_neighbors = torch.einsum("nc,nkc->nk", [q, k_neighbors])

        # Average over valid neighbors
        l_pos_neighbors = l_pos_neighbors.sum(dim=1, keepdim=True) / valid_neighbors_count.unsqueeze(1).clamp(min=1)

        # Negative logits
        l_neg = torch.einsum("nc,ck->nk", [q, queue.clone().detach().T])

        # Combine logits and apply temperature scaling
        logits_neighbors = torch.cat([l_pos_neighbors, l_neg], dim=1)
        logits_neighbors /= self.temperature

        # Ground truth labels (positive key is at index 0)
        labels = torch.zeros(logits_neighbors.shape[0], dtype=torch.long, device=logits_neighbors.device)

        # Cross entropy loss
        loss = F.cross_entropy(logits_neighbors, labels)
        return loss

    def forward(self, q, k_center, k_neighbors, queue, valid_neighbors_count):
        """
        Compute the multi-head contrastive loss.

        Parameters:
        - q (torch.Tensor): Query embeddings.
        - k_center (torch.Tensor): Center key embeddings.
        - k_neighbors (torch.Tensor): Neighbor key embeddings.
        - queue (torch.Tensor): Negative key embeddings.
        - valid_neighbors_count (torch.Tensor): Number of valid neighbors.

        Returns:
        - loss (torch.Tensor): Combined multi-head contrastive loss.
        """
        # Center loss
        loss_center = self.loss_orginal(q, k_center, queue)

        # Neighbor loss
        loss_neighbors = self.loss_neighbor(q, k_neighbors, queue, valid_neighbors_count)

        # Combine losses
        loss = self.alpha * loss_center + (1 - self.alpha) * loss_neighbors
        return loss
