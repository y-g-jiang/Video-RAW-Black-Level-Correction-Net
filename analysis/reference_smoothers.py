"""Seven-frame post-processors for fixed single-frame CFA offset estimates."""

from itertools import product

import numpy as np


WINDOW = 7
CENTER = 3


def centered_windows(values, window=WINDOW):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("values must have shape (frames, channels)")
    if window <= 0 or window % 2 != 1:
        raise ValueError("window must be a positive odd integer")
    if values.shape[0] < window:
        raise ValueError("sequence is shorter than the smoothing window")
    return np.stack([values[i:i + window] for i in range(values.shape[0] - window + 1)])


def mean_smoother(values, window=WINDOW):
    return centered_windows(values, window).mean(axis=1)


def median_smoother(values, window=WINDOW):
    return np.median(centered_windows(values, window), axis=1)


def exponential_weights(rho, window=WINDOW):
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    center = window // 2
    weights = rho ** np.abs(np.arange(window) - center)
    if rho == 0.0:
        weights[:] = 0.0
        weights[center] = 1.0
    return weights / weights.sum()


def exponential_smoother(values, rho, window=WINDOW):
    windows = centered_windows(values, window)
    return np.einsum("tfc,f->tc", windows, exponential_weights(rho, window))


def _measurement_variance(value, channels):
    r = np.asarray(value, dtype=np.float64)
    if r.ndim == 0:
        r = np.full(channels, float(r))
    if r.shape != (channels,) or not np.isfinite(r).all() or np.any(r <= 0):
        raise ValueError("measurement_variance must be positive scalar or one value per channel")
    return r


def _random_walk_covariance(q_steps, p0):
    n = len(q_steps) + 1
    cumulative = np.concatenate(([0.0], np.cumsum(q_steps)))
    indices = np.minimum.outer(np.arange(n), np.arange(n))
    return p0 + cumulative[indices]


def kalman_center(window, q_over_r, measurement_variance=1.0, diffuse_prior_ratio=1e6):
    window = np.asarray(window, dtype=np.float64)
    if window.ndim != 2 or window.shape[0] != WINDOW:
        raise ValueError("window must have shape (7, channels)")
    channels = window.shape[1]
    r_values = _measurement_variance(measurement_variance, channels)
    result = np.empty(channels, dtype=np.float64)
    for c, r in enumerate(r_values):
        covariance = _random_walk_covariance(
            np.full(WINDOW - 1, q_over_r * r), diffuse_prior_ratio * r
        )
        observed_covariance = covariance + np.eye(WINDOW) * r
        center_weights = np.linalg.solve(observed_covariance, covariance[CENTER])
        result[c] = center_weights @ window[:, c]
    return result


def kalman_smoother(values, q_over_r, measurement_variance=1.0):
    windows = centered_windows(values)
    return np.stack([
        kalman_center(w, q_over_r, measurement_variance) for w in windows
    ])


def jump_center(window, q_over_r, jump_probability, jump_variance_over_r,
                measurement_variance=1.0, diffuse_prior_ratio=1e6):
    """Exact Bernoulli-Gaussian fixed-interval posterior mean for a 7-frame window."""
    window = np.asarray(window, dtype=np.float64)
    if window.ndim != 2 or window.shape[0] != WINDOW:
        raise ValueError("window must have shape (7, channels)")
    if not 0.0 <= jump_probability <= 1.0:
        raise ValueError("jump_probability must be in [0, 1]")
    if q_over_r < 0.0 or jump_variance_over_r < 0.0:
        raise ValueError("variance ratios must be nonnegative")

    channels = window.shape[1]
    r_values = _measurement_variance(measurement_variance, channels)
    result = np.empty(channels, dtype=np.float64)
    patterns = list(product((0, 1), repeat=WINDOW - 1))

    for c, r in enumerate(r_values):
        path_means = []
        path_log_weights = []
        z = window[:, c]
        for pattern in patterns:
            jumps = sum(pattern)
            if jump_probability == 0.0 and jumps:
                continue
            if jump_probability == 1.0 and jumps != WINDOW - 1:
                continue
            log_prior = 0.0
            if jump_probability > 0.0:
                log_prior += jumps * np.log(jump_probability)
            if jump_probability < 1.0:
                log_prior += (WINDOW - 1 - jumps) * np.log1p(-jump_probability)

            q_steps = np.asarray([
                (q_over_r + flag * jump_variance_over_r) * r for flag in pattern
            ])
            state_covariance = _random_walk_covariance(
                q_steps, diffuse_prior_ratio * r
            )
            observation_covariance = state_covariance + np.eye(WINDOW) * r
            sign, logdet = np.linalg.slogdet(observation_covariance)
            if sign <= 0:
                raise FloatingPointError("non-positive observation covariance")
            solved = np.linalg.solve(observation_covariance, z)
            log_likelihood = -0.5 * (
                logdet + z @ solved + WINDOW * np.log(2.0 * np.pi)
            )
            path_means.append(state_covariance[CENTER] @ solved)
            path_log_weights.append(log_prior + log_likelihood)

        log_weights = np.asarray(path_log_weights)
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= weights.sum()
        result[c] = weights @ np.asarray(path_means)
    return result


def jump_smoother(values, q_over_r, jump_probability, jump_variance_over_r,
                  measurement_variance=1.0):
    windows = centered_windows(values)
    return np.stack([
        jump_center(w, q_over_r, jump_probability, jump_variance_over_r,
                    measurement_variance)
        for w in windows
    ])
