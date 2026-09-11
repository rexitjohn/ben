from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "age_years", "sample_quality", "volatile_signature_1", "volatile_signature_2",
    "inflammatory_index", "metabolic_index", "oxidative_index", "protein_signal_1",
    "protein_signal_2", "spectral_ratio",
]
TARGET_COLUMN = "cancer_positive"

def make_synthetic_canine_dataset(rows: int = 1000, positive_rate: float = 0.50, random_state: int = 42) -> pd.DataFrame:
    """Create synthetic data for pipeline testing only; not validated biomarkers."""
    if rows < 100:
        raise ValueError("rows must be at least 100")
    if not 0.05 <= positive_rate <= 0.95:
        raise ValueError("positive_rate must be between 0.05 and 0.95")
    rng = np.random.default_rng(random_state)
    y = rng.binomial(1, positive_rate, size=rows)
    age = np.clip(rng.normal(8.0 + 1.2 * y, 3.0, rows), 0.5, 18.0)
    quality = np.clip(rng.beta(8, 2, rows), 0.05, 1.0)
    latent = rng.normal(0, 1, rows) + 1.15 * y
    inflammation = rng.normal(0, 1, rows) + 0.60 * y
    metabolism = rng.normal(0, 1, rows) + 0.45 * y
    oxidative = rng.normal(0, 1, rows) + 0.55 * y
    df = pd.DataFrame({
        "age_years": age,
        "sample_quality": quality,
        "volatile_signature_1": latent + rng.normal(0, 0.75, rows),
        "volatile_signature_2": 0.7 * latent + rng.normal(0, 0.95, rows),
        "inflammatory_index": inflammation,
        "metabolic_index": metabolism,
        "oxidative_index": oxidative,
        "protein_signal_1": 0.6 * inflammation + rng.normal(0, 0.9, rows),
        "protein_signal_2": 0.5 * metabolism + rng.normal(0, 0.9, rows),
        "spectral_ratio": 0.5 * latent + 0.35 * oxidative + rng.normal(0, 0.8, rows),
        "cancer_positive": y.astype(int),
    })
    for col in FEATURE_COLUMNS:
        mask = rng.random(rows) < 0.015
        df.loc[mask, col] = np.nan
    return df
