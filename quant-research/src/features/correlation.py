"""
src/features/correlation.py

Volatility and dynamic correlation features.

Classes:
    GARCHModel   — fits GARCH(1,1) per instrument, exposes conditional sigmas
    DCCGARCH     — computes dynamic conditional correlations across instruments

Standalone functions:
    get_correlation  — rolling pairwise correlation between two series
    get_volatility   — rolling standard deviation (realised vol proxy)
"""

import numpy as np
import pandas as pd
from arch import arch_model


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def get_correlation(series1: pd.Series, series2: pd.Series, past_period: int = 10) -> pd.Series:
    """Rolling Pearson correlation between two return series.

    Args:
        series1: First return series (pandas Series, same index as series2).
        series2: Second return series.
        past_period: Rolling window length in observations.

    Returns:
        Rolling correlation series; inf values replaced with NaN.
    """
    rolling_corr = series1.rolling(window=past_period).corr(series2)
    rolling_corr.replace([np.inf, -np.inf], np.nan, inplace=True)
    return rolling_corr


def get_volatility(returns: pd.Series, past_period: int = 10) -> pd.Series:
    """Rolling realised volatility (std of returns).

    Args:
        returns: Return series.
        past_period: Rolling window length in observations.

    Returns:
        Rolling standard deviation series.
    """
    return returns.rolling(window=past_period).std()


# ---------------------------------------------------------------------------
# GARCH(1,1) per instrument
# ---------------------------------------------------------------------------

class GARCHModel:
    """Fits an independent GARCH(1,1) model for each instrument.

    After calling fit_models(), conditional volatility (sigma) for each
    instrument is available in self.sigmas as a DataFrame.

    Args:
        returns_scaled: DataFrame whose columns are named
                        '<Instrument>ReturnEMA' (e.g. 'BrazilReturnEMA').

    Example:
        >>> garch = GARCHModel(prices[['BrazilReturnEMA', 'ColumbiaReturnEMA']])
        >>> garch.fit_models()
        >>> sigmas = garch.sigmas   # DataFrame of conditional vols
    """

    def __init__(self, returns_scaled: pd.DataFrame):
        self.returns_scaled = returns_scaled
        self.sigmas = pd.DataFrame()

    def fit_models(self, instruments: list[str] | None = None) -> None:
        """Fit GARCH(1,1) for each instrument column.

        Args:
            instruments: List of instrument names (e.g. ['Brazil', 'Columbia']).
                         Defaults to inferring from column names ending in 'ReturnEMA'.
        """
        if instruments is None:
            instruments = [
                col.replace("ReturnEMA", "")
                for col in self.returns_scaled.columns
                if col.endswith("ReturnEMA")
            ]

        for instr in instruments:
            col = instr + "ReturnEMA"
            returns = self.returns_scaled[col]
            notna_mask = returns.notna()

            model = arch_model(returns[notna_mask], vol="Garch", p=1, q=1)
            result = model.fit(disp="off")

            # Reindex back to full index (NaN where data was missing)
            sigma_full = pd.Series(np.nan, index=returns.index)
            sigma_full[notna_mask] = result.conditional_volatility.values
            self.sigmas[col] = sigma_full

        self.sigmas.index = self.returns_scaled.index


# ---------------------------------------------------------------------------
# DCC-GARCH dynamic correlations
# ---------------------------------------------------------------------------

class DCCGARCH:
    """Computes Dynamic Conditional Correlations (DCC) from GARCH sigmas.

    Uses a rolling-window approach to produce pairwise correlations and a
    first-PCA composite across all instruments.

    Args:
        sigmas: DataFrame of GARCH conditional volatilities (from GARCHModel).
        returns_scaled: DataFrame of raw EMA returns (same columns as sigmas).
        instrs: List of instrument names, e.g. ['Brazil', 'Columbia', 'Chile'].

    Example:
        >>> dcc = DCCGARCH(garch.sigmas, prices[return_cols], instruments)
        >>> dcc.calculate_dynamic_correlation(window=10)
        >>> prices['DCC_Correlations'] = [np.nan]*10 + list(dcc.dcc_correlations_1PCA)
    """

    def __init__(
        self,
        sigmas: pd.DataFrame,
        returns_scaled: pd.DataFrame,
        instrs: list[str],
    ):
        self.sigmas = sigmas
        self.returns_scaled = returns_scaled
        self.instrs = instrs

        # Outputs — populated by calculate_dynamic_correlation()
        self.dcc_correlations_1PCA = None
        self.dcc_correlations_pair: dict[tuple[str, str], list[float]] = {}

    def calculate_dynamic_correlation(self, window: int = 30) -> None:
        """Compute rolling DCC correlations.

        Standardises returns by GARCH sigma, then computes the full
        correlation matrix over a rolling window. Pairwise correlations and
        the first PCA eigenvalue (regime indicator) are stored as attributes.

        Args:
            window: Rolling window length in observations.
        """
        n = len(self.instrs)
        pca_1st_eigenvalues: list[float] = []

        # Initialise pairwise storage for all unique pairs
        pairs = [
            (self.instrs[i], self.instrs[j])
            for i in range(n)
            for j in range(i + 1, n)
        ]
        pair_lists: dict[tuple[str, str], list[float]] = {p: [] for p in pairs}

        for i in range(window, len(self.returns_scaled)):
            subset_returns = self.returns_scaled.iloc[i - window:i]
            subset_sigmas = self.sigmas.iloc[i - window:i]
            standardized_returns = subset_returns / subset_sigmas

            # Drop rows with any NaN before computing correlation
            clean = standardized_returns.dropna()
            corr_matrix = np.corrcoef(clean.T)

            # Store pairwise correlations
            col_names = list(subset_returns.columns)
            for (instr_a, instr_b) in pairs:
                idx_a = col_names.index(instr_a + "ReturnEMA")
                idx_b = col_names.index(instr_b + "ReturnEMA")
                pair_lists[(instr_a, instr_b)].append(corr_matrix[idx_a, idx_b])

            # First PCA eigenvalue as regime signal
            if np.isnan(corr_matrix).any():
                pca_1st_eigenvalues.append(np.nan)
            else:
                eigenvalues, _ = np.linalg.eigh(corr_matrix)
                pca_1st_eigenvalues.append(float(np.max(eigenvalues)))

        self.dcc_correlations_1PCA = pca_1st_eigenvalues
        self.dcc_correlations_pair = pair_lists
