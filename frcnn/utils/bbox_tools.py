
import numpy as np
import torch

def loc2bbox(src_bbox, loc):
    """
    Decode bounding box predictions from regression offsets.

    Args:
        src_bbox (torch.Tensor): Source bounding boxes (anchors or proposals) in (y1, x1, y2, x2) format.
        loc (torch.Tensor): Predicted regression offsets (dy, dx, dh, dw).

    Returns:
        torch.Tensor: Decoded bounding boxes in (y1, x1, y2, x2) format.
    """
    if src_bbox.shape[0] == 0:
        return torch.zeros((0, 4), dtype=loc.dtype, device=loc.device)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    ctr_y = dy * src_height[:, None] + src_ctr_y[:, None]
    ctr_x = dx * src_width[:, None] + src_ctr_x[:, None]
    h = torch.exp(dh) * src_height[:, None]
    w = torch.exp(dw) * src_width[:, None]

    dst_bbox = torch.zeros(loc.shape, dtype=loc.dtype, device=loc.device)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_y + 0.5 * w

    return dst_bbox


def bbox2loc(src_bbox, dst_bbox):
    """
    Encode bounding box pairs into regression offsets.

    Args:
        src_bbox (torch.Tensor): Source bounding boxes (anchors or proposals) in (y1, x1, y2, x2) format.
        dst_bbox (torch.Tensor): Target bounding boxes (ground truth) in (y1, x1, y2, x2) format.

    Returns:
        torch.Tensor: Encoded regression offsets (dy, dx, dh, dw).
    """
    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    dst_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    dst_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    dst_ctr_y = dst_bbox[:, 0] + 0.5 * dst_height
    dst_ctr_x = dst_bbox[:, 1] + 0.5 * dst_width

    dy = (dst_ctr_y - src_ctr_y) / src_height
    dx = (dst_ctr_x - src_ctr_x) / src_width
    dh = torch.log(dst_height / src_height)
    dw = torch.log(dst_width / src_width)

    loc = torch.stack((dy, dx, dh, dw), dim=1)
    return loc


def bbox_iou(bbox_a, bbox_b):
    """
    Calculate Intersection over Union (IoU) of bounding boxes.

    Args:
        bbox_a (torch.Tensor): Bounding boxes A in (y1, x1, y2, x2) format.
        bbox_b (torch.Tensor): Bounding boxes B in (y1, x1, y2, x2) format.

    Returns:
        torch.Tensor: IoU values for each pair of bounding boxes.
    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = torch.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = torch.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = torch.prod(br - tl, dim=2) * (tl < br).all(dim=2)
    area_a = torch.prod(bbox_a[:, 2:] - bbox_a[:, :2], dim=1)
    area_b = torch.prod(bbox_b[:, 2:] - bbox_b[:, :2], dim=1)
    iou = area_i / (area_a[:, None] + area_b - area_i)

    return iou


if __name__ == '__main__':
    print("--- Testing Bounding Box Tools ---")

    # Test loc2bbox
    src_bbox = torch.tensor([[10., 10., 50., 50.]])
    loc = torch.tensor([[0.1, 0.1, 0.2, 0.2]])
    decoded_bbox = loc2bbox(src_bbox, loc)
    print(f"Decoded BBox: {decoded_bbox}")
    # Expected: roughly [[14. 14. 54. 54.]]

    # Test bbox2loc
    dst_bbox = torch.tensor([[14., 14., 54., 54.]])
    encoded_loc = bbox2loc(src_bbox, dst_bbox)
    print(f"Encoded Loc: {encoded_loc}")
    # Expected: roughly [[0.1 0.1 0.2 0.2]]

    # Test bbox_iou
    bbox_a = torch.tensor([[0., 0., 10., 10.], [5., 5., 15., 15.]])
    bbox_b = torch.tensor([[0., 0., 10., 10.], [10., 10., 20., 20.]])
    iou_matrix = bbox_iou(bbox_a, bbox_b)
    print(f"IoU Matrix:\n{iou_matrix}")
    # Expected: [[1.   0.14285714]
    #            [0.14285714 0.14285714]]

