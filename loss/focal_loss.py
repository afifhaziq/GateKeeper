import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiFocalLoss(nn.Module):
    def __init__(self, num_class, alpha=None, gamma=2, reduction='mean'):
        """
        Multi-class Focal Loss implementation.
        
        Parameters:
        - alpha: (list or None) A list of weights for each class. If None, all classes are equally weighted.
        - gamma: (float) Focusing parameter to reduce the relative loss for well-classified examples.
        - reduction: (str) Specifies the reduction to apply to the output: 'none', 'mean', 'sum'.
        """
        super(MultiFocalLoss, self).__init__()
        self.num_classs = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        

    def forward(self, logits, targets):
        """
        Compute the focal loss for multi-class classification.
        
        Parameters:
        - logits: (Tensor) Raw model outputs of shape (batch_size, num_classes).
        - targets: (Tensor) Ground truth labels of shape (batch_size,).
        
        Returns:
        - loss: (Tensor) Calculated focal loss.
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)
        
        # Select probabilities for the target class
        batch_size, num_classes = logits.size()
        targets_one_hot = F.one_hot(targets, num_classes).to(logits.device)  # Shape: (batch_size, num_classes)
        
        # Get the probabilities of the true classes
        pt = torch.sum(probs * targets_one_hot, dim=1)  # Shape: (batch_size,)
        
        # Compute the focal loss
        focal_weight = (1 - pt) ** self.gamma
        loss = -focal_weight * torch.log(pt)
        
        # Apply class weights (alpha) if provided
        if self.alpha is not None:
            alpha_t = torch.tensor(self.alpha).to(logits.device)  # Ensure alpha matches device
            alpha_weight = torch.sum(alpha_t * targets_one_hot, dim=1)
            loss *= alpha_weight

        # Reduce the loss if required
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss