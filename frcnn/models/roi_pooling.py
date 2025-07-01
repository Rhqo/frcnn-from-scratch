
import torch
import torch.nn as nn
import numpy as np


class RoIPooling(nn.Module):
    """
    Region of Interest (RoI) Pooling layer.

    This layer takes a feature map and a set of proposals and extracts a
    fixed-size feature vector for each proposal.

    Args:
        output_size (tuple): The desired output size (height, width).
        spatial_scale (float): The scaling factor to map proposal coordinates
                               from the input image space to the feature map space.
    """

    def __init__(self, output_size, spatial_scale):
        super(RoIPooling, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, features, rois):
        """
        Forward pass of the RoI Pooling layer.

        Args:
            features (torch.Tensor): The input feature map of shape 
                                     (batch_size, channels, height, width).
            rois (torch.Tensor): The region proposals of shape (n_rois, 5).
                                 Each row is (batch_index, x1, y1, x2, y2).

        Returns:
            torch.Tensor: The RoI-pooled features of shape 
                          (n_rois, channels, output_height, output_width).
        """
        output = []
        rois = rois.long()

        for i in range(rois.size(0)):
            roi = rois[i]
            batch_idx = roi[0]
            roi_start_w = roi[1].item() * self.spatial_scale
            roi_start_h = roi[2].item() * self.spatial_scale
            roi_end_w = roi[3].item() * self.spatial_scale
            roi_end_h = roi[4].item() * self.spatial_scale

            roi_height = max(roi_end_h - roi_start_h, 1)
            roi_width = max(roi_end_w - roi_start_w, 1)

            bin_size_h = roi_height / self.output_size[0]
            bin_size_w = roi_width / self.output_size[1]

            feature = features[batch_idx]

            pooled_feature = torch.zeros(feature.size(0), self.output_size[0], self.output_size[1]).to(features.device)

            for h in range(self.output_size[0]):
                for w in range(self.output_size[1]):
                    h_start = int(np.floor(h * bin_size_h + roi_start_h))
                    h_end = int(np.ceil((h + 1) * bin_size_h + roi_start_h))
                    w_start = int(np.floor(w * bin_size_w + roi_start_w))
                    w_end = int(np.ceil((w + 1) * bin_size_w + roi_start_w))

                    h_start = min(max(h_start, 0), feature.size(1))
                    h_end = min(max(h_end, 0), feature.size(1))
                    w_start = min(max(w_start, 0), feature.size(2))
                    w_end = min(max(w_end, 0), feature.size(2))

                    if h_end > h_start and w_end > w_start:
                        roi_feature_area = feature[:, h_start:h_end, w_start:w_end]
                        pooled_feature[:, h, w] = torch.max(torch.max(roi_feature_area, dim=2)[0], dim=1)[0]

            output.append(pooled_feature)

        return torch.stack(output)
