
# Utility-Induced Auto-Bidding
## Installation

Use Python 3.8 or newer. From the repository root, install the dependencies:

```bash
python -m pip install "numpy>=1.20" "pandas>=1.3"
```

## Repository structure

```text
docs_auto_bidding/
  README.md
  auto_bidding/
    __init__.py
    strategies/
      __init__.py
      base.py              # Common strategy interface
      pid.py               # PID baseline
      utility_induced.py   # Utility-induced strategy
    data_loader.py         # CSV data loader
    environment.py         # Auction simulator
```

## Quick start

Run this example from the repository root. It computes bids for the first
timestep using synthetic predicted conversion probabilities and does not
require a dataset.

```python
import numpy as np

from auto_bidding.strategies.utility_induced import (
    ScoreMarginalBiddingStrategy,
)

strategy = ScoreMarginalBiddingStrategy(
    budget=100, cpa=2, category=1, variant="sparse"
)
strategy._total_steps = 48  # Replace with the actual episode length.
pvalues = np.array([0.01, 0.02, 0.05], dtype=np.float32)
bids, cpa_dual = strategy.bidding(
    timeStepIndex=0,
    pValues=pvalues,
    pValueSigmas=np.zeros_like(pvalues),
    historyPValueInfo=[],
    historyBid=[],
    historyAuctionResult=[],
    historyImpressionResult=[],
    historyLeastWinningCost=[],
)
print("Bids:", bids)
print("CPA dual:", cpa_dual)
```

Select `variant="dense"` for dense data. To construct the baseline, import
`PidCpaBiddingStrategy` from
`auto_bidding.strategies.pid` and supply the
campaign's `budget`, `cpa`, and `category`.

## Fixed configurations

| Parameter | Dense | Sparse |
| --- | --- | --- |
| beta | 0.0 | 0.0 |
| gamma | 1.8 | 1.8 |
| Relative temperature | 0.005 | 0.005 |
| Conversion feedback | Forecast | Forecast |
| Dual update frequency | Every timestep | Every timestep |
| Dual step size | 0.0001 | 0.01 |
| Initial CPA dual / budget dual | 0 / 1 | 0 / 1 |
| RoS control bound U | 1.0 | 1.0 |
| Gradient scale n_ref | 1.0 | 1.0 |
| Dual projection cap | 1000 | 1000 |
| Bid cap | 1000000 | 1000000 |

The three utility pieces are `(1, 0)`, `(1, 0)`, and `(2.8, -1.8)`.
Both identical first pieces are retained in the soft-min weights. The CPA
ratio uses the existing cold-start offsets of 2 for normalized payment
and forecast conversions.

The PID baseline uses `kp=0.1`, `ki=0.01`, `kd=0.03`, minimum multiplier
`0.01`, maximum multiplier `3 * cpa`, integral clip `50`, CPA EMA weight
`0.5`, and minimum conversion count `1e-6`.

## Feedback and episode integration

Set `_total_steps` to the episode length. Before each call to `bidding`,
update `remaining_budget` from the evaluator's cash ledger. After each
auction, append the timestep arrays to the histories:

- `historyPValueInfo`: rows contain predicted conversion probability and
  prediction uncertainty.
- `historyAuctionResult`: rows contain win status, exposure status, and
  cash payment, in that order.
- `historyImpressionResult`: the first column contains realized conversions
  for the PID baseline.

The utility-induced strategy reads the previous timestep's forecast
conversions as `sum(pValue * win)` and cash payments from the auction
history. The CPA ratio uses cold-start offsets of 2, as described above.
The caller enforces the aggregate timestep budget when executing bids;
the strategy itself caps each bid by the available campaign budget.
Call `reset()` before starting a new episode and set its episode length.

## Data availability

Traffic data and a complete evaluation runner are supplied separately.
This branch contains no downloaded datasets or automatic download script.
The quick-start example verifies strategy invocation only.

Load an externally prepared traffic CSV with:

```python
from auto_bidding.data_loader import TestDataLoader

loader = TestDataLoader(file_path="/path/to/traffic/period-7.csv")
episode = loader.mock_data(loader.keys[0])
```

The CSV must contain `deliveryPeriodIndex`, `advertiserNumber`,
`timeStepIndex`, `pValue`, `pValueSigma`, `leastWinningCost`, `budget`,
`CPAConstraint`, and `advertiserCategoryIndex`. Each returned episode
contains its timestep count, predicted values, uncertainties, market
prices, budget, target CPA, and category.

The loader writes `raw_data.pickle` next to the CSV and reuses that cache
on later loads. Use a separate directory for each traffic CSV or remove
the old cache before switching files.
