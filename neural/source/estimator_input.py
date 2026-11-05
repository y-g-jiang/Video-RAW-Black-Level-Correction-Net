"""Estimator-only fixed window. DN units and source RAW are preserved.

The caller subtracts its own nominal black before this adapter. Do not
replace source-camera PTC span by 4095-512. Do not round floating input.
Apply this once before ALL learned/statistical image feature branches.
"""
import numpy as np

PEDESTAL_DN = 512.0
WHITE_CODE = 4095.0
LOWER_DN = -PEDESTAL_DN
UPPER_DN = WHITE_CODE - PEDESTAL_DN


def estimator_view(signed_native_dn):
    """Accept NumPy arrays or torch tensors, preserving geometry and DN scale.

    Tensor clamp is out-of-place; every branch must consume this return value.
    No changes to WB, noise labels, bit depth, or the original RAW buffer.
    """
    if isinstance(signed_native_dn, np.ndarray):
        if not np.issubdtype(signed_native_dn.dtype, np.floating):
            raise TypeError('Subtract nominal black in floating point first')
        return np.clip(signed_native_dn, LOWER_DN, UPPER_DN)
    if not signed_native_dn.is_floating_point():
        raise TypeError('Subtract nominal black in floating point first')
    return signed_native_dn.clamp(min=LOWER_DN, max=UPPER_DN)


def clipping_rates(signed_native_dn):
    """CPU diagnostics only; do not feed saturation rates or PTC to the model."""
    a = np.asarray(signed_native_dn)
    return {'lower_fraction': float(np.mean(a < LOWER_DN)),
            'upper_fraction': float(np.mean(a > UPPER_DN))}


def correct_original(original_signed_dn, predicted_bias_dn):
    """Subtract bias from the ORIGINAL full-range signed RAW; no input clamp."""
    return original_signed_dn - predicted_bias_dn
