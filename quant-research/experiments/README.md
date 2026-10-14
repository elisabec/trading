# Experiments

The lab notebook: one row per run (strategy, parameters, period, IC, Sharpe, drawdown, notes).

A **run** is one backtest with one set of parameters. A **grid search** creates many runs at once (e.g. 4 deltas x 3 hedge budgets = 12 runs); all of them land here, not only the best one.

Keeps track of how many variants were tried, the main defence against overfitting.
