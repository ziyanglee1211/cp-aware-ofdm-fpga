"""Descriptive, slot-clustered statistics for the 2026-10 research round.

Within one checkpoint/channel/SNR, error counts must be paired by coherence
slot. Bits inside a slot are NOT independent resampling units. Percentile
bootstrap intervals condition on that checkpoint and simulated channel law;
they neither include training-seed uncertainty nor adjust multiple screening
comparisons. No hypothesis tests or claims of statistical significance.
"""
import math

import numpy as np


def summarize_paired_slots(candidate_errors, reference_errors, bits_per_slot,
                           bootstrap_reps=2000, bootstrap_seed=20261001):
    """Return candidate-minus-reference BER and a conditional 95% interval.

    Inputs are one-dimensional integer error counts per common received slot.
    For QPSK the denominator includes both bits per data resource element.
    Negative delta means fewer errors for the candidate, not a guaranteed
    improvement under retraining or hardware implementation.
    """
    a = np.asarray(candidate_errors)
    b = np.asarray(reference_errors)
    if a.ndim != 1 or b.shape != a.shape or len(a) < 2:
        raise ValueError('Require >=2 paired slots in equal 1D arrays')
    if isinstance(bits_per_slot, bool) or int(bits_per_slot) != bits_per_slot or bits_per_slot <= 0:
        raise ValueError('bits_per_slot must be a positive integer')
    for values in (a, b):
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            raise ValueError('Error counts must be finite integers')
        if np.any(values < 0) or np.any(values > bits_per_slot):
            raise ValueError('Error count outside slot bit denominator')
    if isinstance(bootstrap_reps, bool) or int(bootstrap_reps) != bootstrap_reps or bootstrap_reps < 0:
        raise ValueError('bootstrap_reps must be a nonnegative integer')
    delta = (a.astype(np.float64) - b.astype(np.float64)) / int(bits_per_slot)
    ci = None
    if bootstrap_reps:
        rng = np.random.default_rng(bootstrap_seed)
        means = np.empty(int(bootstrap_reps), dtype=np.float64)
        for start in range(0, int(bootstrap_reps), 128):
            stop = min(start + 128, int(bootstrap_reps))
            sampled = rng.integers(0, len(delta), size=(stop - start, len(delta)))
            means[start:stop] = delta[sampled].mean(axis=1)
        lo, hi = np.quantile(means, [0.025, 0.975])
        ci = [float(lo), float(hi)]
    return {
        'slots': int(len(a)), 'bits_per_slot': int(bits_per_slot),
        'candidate_error_bits': int(a.sum()), 'reference_error_bits': int(b.sum()),
        'candidate_ber': float(a.sum() / (len(a) * bits_per_slot)),
        'reference_ber': float(b.sum() / (len(a) * bits_per_slot)),
        'candidate_minus_reference_ber': float(delta.mean()),
        'slot_delta_sample_sd': float(delta.std(ddof=1)),
        'conditional_slot_bootstrap_95pct_ci': ci,
        'bootstrap_reps': int(bootstrap_reps), 'bootstrap_seed': int(bootstrap_seed),
        'resampling_unit': 'one common simulated coherence slot',
        'scope': 'Exploratory descriptive interval conditional on checkpoint; '
                 'not training-seed uncertainty or multiplicity-adjusted inference.',
    }


def summarize_seed_deltas(deltas):
    """Show every independent training-seed contrast and sample SD, no p value."""
    values = [float(value) for value in deltas]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError('Require at least one finite seed delta; censoring must be explicit')
    array = np.asarray(values, dtype=np.float64)
    return {'training_seeds': len(values), 'raw_seed_deltas': values,
            'mean_delta': float(array.mean()),
            'sample_sd': float(array.std(ddof=1)) if len(array) > 1 else None,
            'uncertainty_scope': 'Between-training-seed descriptive spread; no hypothesis test'}
