import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_resnet_weights(base_encoder):
    """
    Get the appropriate ResNet weights class based on the base_encoder string.

    Parameters:
    - base_encoder (str): The name of the ResNet model (e.g., "resnet18", "resnet50").

    Returns:
    - weights (torchvision.models.weights): The corresponding ResNet weights class.
    """
    resnet_weights_map = {
        "resnet18": ResNet18_Weights.DEFAULT,
        "resnet34": ResNet34_Weights.DEFAULT,
        "resnet50": ResNet50_Weights.DEFAULT,
        "resnet101": ResNet101_Weights.DEFAULT,
        "resnet152": ResNet152_Weights.DEFAULT,
    }
    
    # Ensure base_encoder is lowercase
    base_encoder = base_encoder.lower()
    
    # Retrieve weights based on base_encoder
    if base_encoder in resnet_weights_map:
        return resnet_weights_map[base_encoder]
    else:
        raise ValueError(f"Unsupported base_encoder: {base_encoder}. Supported values are: {list(resnet_weights_map.keys())}")

class MoCoV2Encoder(nn.Module):
    def __init__(self, base_encoder='resnet50', output_dim=128, queue_size=65536, momentum=0.999, temperature=0.07):
        """
        Initialize MoCoV2Encoder with a specified ResNet backbone.

        Parameters:
        - base_encoder: str, specifies the ResNet variant to use ('resnet18', 'resnet34', 'resnet50', etc.)
        - output_dim: int, the dimension of the output features after the projection head
        - queue_size: int, the size of the queue for negative samples
        - momentum: float, the momentum for updating the key encoder
        - temperature: float, temperature scaling for contrastive loss
        """
        super(MoCoV2Encoder, self).__init__()
        
        # Load encoders and setup projection layers
        self.encoder_q = self._load_resnet(base_encoder, output_dim)
        self.encoder_k = self._load_resnet(base_encoder, output_dim)
        
        self.temperature = temperature
        self.momentum = momentum

        # Initialize queue for negative samples
        self.register_buffer("queue", torch.randn(queue_size, output_dim))  #-- this line
        self.queue = F.normalize(self.queue, dim=1)  # Normalize the queue
        self.queue_ptr = 0

        # Initialize the momentum encoder parameters
        self._initialize_momentum_encoder()
        logger.info(f"MoCoV2Encoder initialized with base encoder: {base_encoder}, output dimension: {output_dim}, queue size: {queue_size}, momentum: {momentum}")

    def _load_resnet(self, base_encoder, output_dim, pretrained=False):
        """
        Load the specified ResNet model and add a projection head.

        Parameters:
        - base_encoder: str, the name of the ResNet model to load
        - output_dim: int, the dimension of the output features after projection head
        
        Returns:
        - encoder: nn.Module, the modified ResNet model with a projection head
        """
        try:
            weight_type = "not default"
            weights=None
            if pretrained:
                weights = get_resnet_weights(base_encoder) # Get weights for the specific ResNet
                weight_type = "default"
                
            encoder = getattr(models, base_encoder)(weights=weights)
            logger.info(f"Using {weight_type} weights of ResNet")
          
            
        except AttributeError:
            raise ValueError(f"Unsupported ResNet variant '{base_encoder}'. Choose from 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'")
        
        # Projector MLP
        hidden_dim = encoder.fc.in_features
        encoder.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        logger.info(f"Loaded {base_encoder} with output dimension: {output_dim}")
        return encoder

    def _initialize_momentum_encoder(self):
        """
        Initialize the momentum encoder by copying parameters from the query encoder.
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Update the momentum encoder parameters by moving average of the query encoder parameters.
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.mul_(self.momentum).add_(param_q.data, alpha=1 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Safely update the queue with the given keys, handling smaller batch sizes.
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Ensure the queue pointer wraps around properly
        end_ptr = ptr + batch_size
        if end_ptr <= self.queue.size(0):
            # Regular case: Enough space in the queue
            self.queue[ptr:end_ptr, :] = keys
        else:
            # Wrap-around case: Split the batch across the queue boundary
            first_part = self.queue.size(0) - ptr
            self.queue[ptr:, :] = keys[:first_part, :]
            self.queue[:end_ptr % self.queue.size(0), :] = keys[first_part:, :]

        # Update the pointer
        self.queue_ptr = (ptr + batch_size) % self.queue.size(0)

    def forward(self, x_q, x_k):
        """
        Forward pass for query and key inputs.
        
        Parameters:
        - x_q: torch.Tensor, input tensor for query
        - x_k: torch.Tensor, input tensor for key
        
        Returns:
        - q: torch.Tensor, normalized feature vector from the query encoder
        - k: torch.Tensor, normalized feature vector from the key encoder
        """
        q = self.encoder_q(x_q)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(x_k)
            k = F.normalize(k, dim=1)
        
        return q, k

    def update_queue(self, keys):
        """
        Public method to update the queue with keys after each forward pass.
        
        Parameters:
        - keys: torch.Tensor, the keys (feature vectors) to add to the queue
        """
        self._dequeue_and_enqueue(keys)