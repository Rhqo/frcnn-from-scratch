

import numpy as np


def generate_base_anchors(base_size=16, ratios=[0.5, 1, 2], scales=2**np.arange(3, 6)):
    """
    Generate base anchors of different scales and aspect ratios.

    Args:
        base_size (int): The base size of the anchors.
        ratios (list): A list of aspect ratios for the anchors.
        scales (np.ndarray): An array of scales for the anchors.

    Returns:
        np.ndarray: An array of shape (len(ratios) * len(scales), 4) representing
                    the base anchors in (y_min, x_min, y_max, x_max) format.
    """
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    w = base_anchor[2] - base_anchor[0] + 1
    h = base_anchor[3] - base_anchor[1] + 1
    x_ctr = base_anchor[0] + 0.5 * (w - 1)
    y_ctr = base_anchor[1] + 0.5 * (h - 1)

    h_ratios = np.sqrt(ratios)
    w_ratios = 1. / h_ratios

    ws = (w * w_ratios[:, np.newaxis] * scales[np.newaxis, :]).flatten()
    hs = (h * h_ratios[:, np.newaxis] * scales[np.newaxis, :]).flatten()

    base_anchors = np.vstack([
        y_ctr - 0.5 * (hs - 1),
        x_ctr - 0.5 * (ws - 1),
        y_ctr + 0.5 * (hs - 1),
        x_ctr + 0.5 * (ws - 1)
    ]).transpose()

    return base_anchors


def generate_anchors(base_anchors, feat_stride, height, width):
    """
    Generate anchors for a feature map of a given size.

    Args:
        base_anchors (np.ndarray): The base anchors.
        feat_stride (int): The stride of the feature map.
        height (int): The height of the feature map.
        width (int): The width of the feature map.

    Returns:
        np.ndarray: An array of shape (height * width * n_anchor, 4) representing
                    all anchors in (y_min, x_min, y_max, x_max) format.
    """
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack([
        shift_y.ravel(),
        shift_x.ravel(),
        shift_y.ravel(),
        shift_x.ravel()
    ]).transpose()

    n_anchor = base_anchors.shape[0]
    n_shifts = shifts.shape[0]
    anchors = (base_anchors.reshape((1, n_anchor, 4)) +
               shifts.reshape((1, n_shifts, 4)).transpose((1, 0, 2)))
    anchors = anchors.reshape((n_shifts * n_anchor, 4))

    return anchors


if __name__ == '__main__':
    print("--- Testing Anchor Generation ---")

    # Test base anchor generation
    base_anchors = generate_base_anchors()
    print(f"Generated {base_anchors.shape[0]} base anchors:")
    print(base_anchors)

    # Test anchor generation for a feature map
    feat_height = 37
    feat_width = 50
    feat_stride = 16
    anchors = generate_anchors(base_anchors, feat_stride, feat_height, feat_width)
    print(f"\nGenerated {anchors.shape[0]} total anchors for a {feat_height}x{feat_width} feature map.")
    print(f"Shape of all anchors: {anchors.shape}")
    print("--- Test Complete ---")

