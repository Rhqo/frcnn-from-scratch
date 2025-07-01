
import torch
import torch.nn as nn


class FastRCNNHead(nn.Module):
    """
    The Fast R-CNN detection head.

    This head takes the RoI-pooled features and performs final classification
    and bounding box regression.

    Args:
        in_channels (int): Number of input channels from the RoI-pooled features.
        num_classes (int): Number of object classes (including background).
        roi_output_size (tuple): The fixed output size of the RoI Pooling layer (height, width).
    """

    def __init__(self, in_channels=2048, num_classes=21, roi_output_size=(7, 7)):
        super(FastRCNNHead, self).__init__()

        # Calculate the input features for the first fully connected layer
        # This is channels * height * width from the RoI-pooled features
        self.in_features = in_channels * roi_output_size[0] * roi_output_size[1]

        # Fully connected layers for classification and regression
        self.fc1 = nn.Linear(self.in_features, 4096)
        self.fc2 = nn.Linear(4096, 4096)

        # Classification layer: outputs scores for each class (including background)
        self.cls_score = nn.Linear(4096, num_classes)

        # Bounding box regression layer: outputs 4 deltas for each class
        self.bbox_pred = nn.Linear(4096, num_classes * 4)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights for the Fast R-CNN head layers.
        """
        nn.init.normal_(self.fc1.weight, 0, 0.01)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, 0, 0.01)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.cls_score.weight, 0, 0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.bbox_pred.weight, 0, 0.001) # Smaller std for bbox regression
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Fast R-CNN head.

        Args:
            x (torch.Tensor): RoI-pooled features of shape 
                              (n_rois, channels, output_height, output_width).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - cls_scores (torch.Tensor): Classification scores for each proposal.
                - bbox_preds (torch.Tensor): Bounding box regression predictions for each proposal.
        """
        # Flatten the pooled features
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers with ReLU activation and dropout
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.dropout(x, p=0.5, training=self.training)

        # Get final classification scores and bounding box predictions
        cls_scores = self.cls_score(x)
        bbox_preds = self.bbox_pred(x)

        return cls_scores, bbox_preds
