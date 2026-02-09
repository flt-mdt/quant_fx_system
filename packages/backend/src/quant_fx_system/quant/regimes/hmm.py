from __future__ import annotations

import numpy as np
import pandas as pd

from .types import HMMConfig


def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    max_val = np.max(a, axis=axis, keepdims=True)
    stable = a - max_val
    sum_exp = np.sum(np.exp(stable), axis=axis, keepdims=True)
    output = max_val + np.log(sum_exp)
    if axis is not None:
        output = np.squeeze(output, axis=axis)
    return output


def _log_gaussian_diag(x: np.ndarray, means: np.ndarray, vars_: np.ndarray) -> np.ndarray:
    n_states, n_features = means.shape
    x = x[:, None, :]
    means = means[None, :, :]
    vars_ = vars_[None, :, :]
    log_det = np.sum(np.log(vars_), axis=2)
    quad = np.sum((x - means) ** 2 / vars_, axis=2)
    return -0.5 * (n_features * np.log(2 * np.pi) + log_det + quad)


def _init_params(features: np.ndarray, cfg: HMMConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_states = cfg.n_states
    n_features = features.shape[1]
    rng = np.random.default_rng(cfg.random_seed)

    if cfg.init_method == "quantile":
        base = features[:, 0]
        quantiles = np.linspace(0, 1, n_states + 1)[1:-1]
        thresholds = np.quantile(base, quantiles)
        bins = [-np.inf, *thresholds, np.inf]
        labels = np.digitize(base, bins) - 1
        means = np.zeros((n_states, n_features))
        vars_ = np.zeros((n_states, n_features))
        for k in range(n_states):
            mask = labels == k
            if not np.any(mask):
                means[k] = np.nanmean(features, axis=0)
                vars_[k] = np.nanvar(features, axis=0)
            else:
                means[k] = np.nanmean(features[mask], axis=0)
                vars_[k] = np.nanvar(features[mask], axis=0)
    else:
        centers = features[rng.choice(len(features), n_states, replace=False)]
        for _ in range(10):
            dists = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            for k in range(n_states):
                mask = labels == k
                if np.any(mask):
                    centers[k] = features[mask].mean(axis=0)
        means = centers
        vars_ = np.zeros((n_states, n_features))
        for k in range(n_states):
            mask = labels == k
            if np.any(mask):
                vars_[k] = np.var(features[mask], axis=0)
            else:
                vars_[k] = np.var(features, axis=0)

    vars_ = np.maximum(vars_, cfg.min_var)
    pi = np.full(n_states, 1.0 / n_states)
    A = np.full((n_states, n_states), 1.0 / n_states)
    return pi, A, means, vars_


def _forward_backward(
    features: np.ndarray,
    pi: np.ndarray,
    A: np.ndarray,
    means: np.ndarray,
    vars_: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    log_A = np.log(A)
    log_pi = np.log(pi)
    log_B = _log_gaussian_diag(features, means, vars_)

    n_samples, n_states = log_B.shape
    log_alpha = np.zeros((n_samples, n_states))
    log_alpha[0] = log_pi + log_B[0]
    for t in range(1, n_samples):
        log_alpha[t] = log_B[t] + _logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0)

    log_beta = np.zeros((n_samples, n_states))
    for t in range(n_samples - 2, -1, -1):
        log_beta[t] = _logsumexp(log_A + log_B[t + 1] + log_beta[t + 1], axis=1)

    loglik = float(_logsumexp(log_alpha[-1], axis=0))
    gamma = np.exp(log_alpha + log_beta - loglik)
    xi_sum = np.zeros((n_states, n_states))
    for t in range(n_samples - 1):
        log_xi = (
            log_alpha[t][:, None]
            + log_A
            + log_B[t + 1][None, :]
            + log_beta[t + 1][None, :]
            - loglik
        )
        xi_sum += np.exp(log_xi)
    return gamma, xi_sum, loglik


def fit_hmm_gaussian(features: pd.DataFrame, cfg: HMMConfig) -> dict:
    data = features.to_numpy()
    pi, A, means, vars_ = _init_params(data, cfg)

    loglik_path = []
    converged = False
    for iteration in range(cfg.max_iter):
        gamma, xi_sum, loglik = _forward_backward(data, pi, A, means, vars_)
        loglik_path.append(loglik)

        pi = gamma[0]
        A = xi_sum / xi_sum.sum(axis=1, keepdims=True)

        weights = gamma.sum(axis=0)
        means = (gamma.T @ data) / weights[:, None]
        diff = data[:, None, :] - means[None, :, :]
        vars_ = (gamma[:, :, None] * diff**2).sum(axis=0) / weights[:, None]
        vars_ = np.maximum(vars_, cfg.min_var)

        if iteration > 0 and abs(loglik_path[-1] - loglik_path[-2]) < cfg.tol:
            converged = True
            break

    return {
        "pi": pi,
        "A": A,
        "means": means,
        "vars": vars_,
        "loglik_path": loglik_path,
        "loglik_final": loglik_path[-1] if loglik_path else None,
        "n_iter": len(loglik_path),
        "converged": converged,
    }


def hmm_filter(features: pd.DataFrame, params: dict) -> pd.DataFrame:
    data = features.to_numpy()
    pi = params["pi"]
    A = params["A"]
    means = params["means"]
    vars_ = params["vars"]

    log_A = np.log(A)
    log_pi = np.log(pi)
    log_B = _log_gaussian_diag(data, means, vars_)

    n_samples, n_states = log_B.shape
    log_alpha = np.zeros((n_samples, n_states))
    log_alpha[0] = log_pi + log_B[0]
    for t in range(1, n_samples):
        log_alpha[t] = log_B[t] + _logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0)
    log_alpha_norm = log_alpha - _logsumexp(log_alpha, axis=1)[:, None]
    probs = np.exp(log_alpha_norm)
    columns = [f"p_state_{i}" for i in range(n_states)]
    return pd.DataFrame(probs, index=features.index, columns=columns)


def hmm_smooth(features: pd.DataFrame, params: dict) -> pd.DataFrame:
    gamma, _, _ = _forward_backward(
        features.to_numpy(), params["pi"], params["A"], params["means"], params["vars"]
    )
    columns = [f"p_state_{i}" for i in range(gamma.shape[1])]
    return pd.DataFrame(gamma, index=features.index, columns=columns)


def hmm_viterbi(features: pd.DataFrame, params: dict) -> pd.Series:
    data = features.to_numpy()
    pi = params["pi"]
    A = params["A"]
    means = params["means"]
    vars_ = params["vars"]

    log_A = np.log(A)
    log_pi = np.log(pi)
    log_B = _log_gaussian_diag(data, means, vars_)

    n_samples, n_states = log_B.shape
    delta = np.zeros((n_samples, n_states))
    psi = np.zeros((n_samples, n_states), dtype=int)
    delta[0] = log_pi + log_B[0]
    for t in range(1, n_samples):
        scores = delta[t - 1][:, None] + log_A
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = log_B[t] + scores[psi[t], range(n_states)]

    states = np.zeros(n_samples, dtype=int)
    states[-1] = int(np.argmax(delta[-1]))
    for t in range(n_samples - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return pd.Series(states, index=features.index, name="regime")


__all__ = ["fit_hmm_gaussian", "hmm_filter", "hmm_smooth", "hmm_viterbi"]
