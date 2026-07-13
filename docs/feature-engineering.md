# Feature Engineering Deep Dive

Why each of the 17 features exists, and the two design constraints that shaped the pipeline: **no future leakage** and **offline/online parity**. All code references are to [`src/features.py`](../src/features.py).

## The two constraints

**No future leakage.** Training data is processed strictly chronologically (`sort_values(by='step')`), and every historical feature for a transaction is computed from a per-user state dictionary that only contains *earlier* transactions. A model that peeks even one step ahead posts beautiful offline metrics and collapses in production.

**Offline/online parity.** The same class exposes `transform` (batch, for training) and `transform_realtime` (single transaction + the sender's recent history pulled from the DB). Both paths compute the same 17 features with the same code, which is the only reliable defense against training/serving skew — the most common way fraud models die in production.

## The features, by what they measure

### Behavioral deviation (the core signal)

| Feature | What it asks |
|---|---|
| `amount_deviation` | Z-score of this amount vs the sender's trailing 30-day history (720 hourly steps). *"Is 45,000 KES normal for this person?"* — for a mama mboga it's a siren, for a wholesaler it's Tuesday. |
| `amount_percentile` | Where this amount ranks in the sender's own recent distribution — robust companion to the z-score when history is short or skewed. |
| `velocity_1h` / `velocity_24h` | Transaction counts in the last hour/day. Account takeover looks like a burst: the thief drains in minutes, not weeks. |
| `time_since_last_transaction` | Dormancy signal. First-ever transactions get a sentinel of `999` — a brand-new or long-dormant account moving money is its own risk category, and the model learns the sentinel as exactly that flag. |

### Transaction shape

| Feature | What it asks |
|---|---|
| `amount_to_balance_ratio` | How much of the account is being moved. Fraud empties accounts; the ratio ≈ 1.0 pattern is the classic drain. |
| `is_round_amount` | Humans buying things pay 1,437 KES; thieves testing limits move 10,000 even. |
| `is_transfer` / `is_cash_out` | In PaySim (and real mobile money), 100% of fraud lives in TRANSFER and CASH_OUT — the compromise→mule→agent-cashout chain. Other types short-circuit to zero risk in the API for throughput. |
| `hour_of_day` | Derived as `step % 24`. Fraud concentrates between midnight and 6 AM — victims are asleep and can't react to the SMS. One of the strongest single features. |

### Counterparty risk

| Feature | What it asks |
|---|---|
| `merchant_risk_score` | Historical fraud rate by destination prefix, **learned in `fit()` from training data only** — it's a lookup table frozen at training time, not recomputed on test data (that would be target leakage). Unknown prefixes fall back to a default rate. |
| `cross_border` | Destination-type flag (merchant-prefixed destinations in PaySim's encoding). |

### Raw context

`amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest` — the model can learn balance-consistency violations (e.g. destination balance not increasing by the sent amount, a known PaySim fraud tell) directly from the raw columns rather than us hand-coding every arithmetic identity.

## What's deliberately absent

- **No graph features** (mule-network detection via destination clustering). Highest-value next addition; requires a graph store and changes the serving architecture.
- **No device/SIM/location signals.** PaySim doesn't have them; a real operator integration would add SIM-swap recency — the single most predictive field in actual mobile-money fraud — device fingerprint, and agent GPS.
- **No categorical embeddings** of account IDs. With millions of accounts, ID embeddings memorize rather than generalize (and make the model stale within weeks).

## The cost profile

The history features make scoring O(sender's 30-day transaction count) per transaction — bounded and cache-friendly. The realtime path pulls one indexed DB query (sender's recent rows) and computes in-process; this is what keeps the API inside its 100 ms budget.
