# FX — DCC-GARCH Mean Reversion Signal

Systematic intraday mean-reversion signal across correlated FX pairs using dynamic conditional correlations.

**Instruments:** Configurable — any set of correlated FX pairs
**Data frequency:** 1-minute mid prices + bid/ask spreads

---

## Hypothesis

Correlated FX pairs exhibit strong intraday co-movement driven by shared macro exposures. When dynamic correlations spike, a temporary dislocation in one pair relative to the others tends to mean-revert within minutes. The signal exploits this by entering in the direction of reversion when correlation confirms a high-co-movement regime.

---

## Methodology

### 1. Volatility Modelling
- Fit a **GARCH(1,1)** model per instrument on 1-min EMA returns
- Produces time-varying conditional volatility (σ) used to standardize returns

### 2. Dynamic Correlation — DCC-GARCH
- Compute **DCC (Dynamic Conditional Correlation)** across instrument pairs
- Rolling window standardized returns → correlation matrix per timestep
- Extract pairwise correlations
- Use first PCA eigenvalue of the correlation matrix as a regime indicator

### 3. Signal Generation
Five signal types tested (configurable via `signalType`):

| Signal Type | Description |
|---|---|
| `dcc` | Average of pairwise DCC correlations per instrument |
| `dccPCA` | First principal component of the full DCC correlation matrix |
| `all` | Average of raw rolling EMA correlations |
| `complement` | Cross-signal: each instrument gets the correlation of the *other two* |
| `random` | Baseline / noise benchmark |

Signal is triggered when `abs(SignalStrength) > corr_threshold`.
Direction (momentum vs reversion) is a configurable parameter.

### 4. Trade Execution & Risk Control
- **Entry:** Risk-inducing trade on signal trigger at best bid/ask
- **Exit:** Trailing stop — closes when PnL retraces >20% from peak, or after minimum holding period
- **Cooldown:** Minimum 5-minute wait between trades per instrument
- **Position:** Fixed notional per trade
- **Spread cost:** Inception PnL = −|position × spread| at entry

### 5. Markout Analysis
Forward returns measured at ±10, ±20, ±30 minutes post-signal to validate edge before running full backtest.

---

## Key Parameters

| Parameter | Description | Best Value Found |
|---|---|---|
| `emapastLook` | EMA lookback window (minutes) | 5 |
| `corr_period` | Rolling correlation window (minutes) | 10 |
| `return_period_direction` | Lookback to determine trade direction (minutes) | 5 |
| `corr_threshold` | Minimum signal strength to trigger trade | 0.6–0.7 |
| `momentumSignal` | `True` = momentum, `False` = mean reversion | `False` |
| `signalType` | Which correlation metric to use as signal | `dcc` |

---

## Backtest Results

Grid search across signal types, correlation thresholds, and momentum/reversion flags. Best configuration:

| Metric | Value |
|---|---|
| Sharpe Ratio (p.a.) | 3.37 |
| Max Drawdown | −3.93 |
| Signal Type | `dcc`, mean reversion |
| Lookback | 50-period, 30 std spacing |

> ⚠️ Results are in-sample. Out-of-sample validation is the next step.

---

## File Structure
research/fx/
├── dcc_signal.py       # Signal implementation
└── README.md           # This file