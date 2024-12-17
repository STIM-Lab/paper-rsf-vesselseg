# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:45:28 2024

@author: meher
"""

import numpy as np
import matplotlib.pyplot as plt
def interpolate_volumes(half_1, half_2, overlap_width):
    """
    Linearly interpolates two 3D volumes over a specified overlapping region.

    Parameters:
    - half_1: First 3D numpy array (shape: Z, Y, X)
    - half_2: Second 3D numpy array (shape: Z, Y, X)
    - overlap_width: The number of pixels in the overlapping region along the Y-axis

    Returns:
    - Merged 3D numpy array
    """

    # Ensure the Z and X dimensions match
    assert half_1.shape[0] == half_2.shape[0], "Mismatch in Z dimension"
    assert half_1.shape[2] == half_2.shape[2], "Mismatch in X dimension"

    # Extract the non-overlapping regions
    half_1_part = half_1[:, :-overlap_width, :]
    half_2_part = half_2[:, overlap_width:, :]

    # Create the interpolation weights
    alpha = np.linspace(0, 1, overlap_width)[None, :, None]  # Shape: (1, overlap_width, 1)

    # Linearly interpolate the overlapping region
    interpolated_area = (1 - alpha) * half_1[:, -overlap_width:, :] + alpha * half_2[:, :overlap_width, :]

    # Concatenate the parts to create the final merged volume
    merged_volume = np.concatenate((half_1_part, interpolated_area, half_2_part), axis=1)

    return merged_volume

a = np.load("1st_half.npy")
b = np.load("2nd_half.npy")

a = np.squeeze(a)
b = np.squeeze(b)
c = interpolate_volumes(a,b,46)

np.save("interpolated.npy", c)