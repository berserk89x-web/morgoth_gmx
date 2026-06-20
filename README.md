# MORGOTH GMX

An autonomous BTC perpetuals trading bot research project
for GMX V2 on Arbitrum. Honest engineering experiment
on the limits of ML-based directional prediction.

## TL;DR

Tested whether machine learning can predict 1-hour BTC 
direction using technical analysis + alternative data 
(funding rates, on-chain flows). Result: **negative**.

Direction at this horizon is not predictable with these 
feature classes. Volatility regimes ARE predictable.
This finding is fully documented across 19 commits.

**This repository demonstrates:**
- Production-grade Python ML pipeline
- Causal feature engineering (no look-ahead bias)
- Honest hypothesis testing
- Walk-forward validation
- Multiple model comparison
- Disciplined git workflow

## What Was Built

### Phase 1 — Foundation
- Arbitrum RPC verified (works from Algeria)
- Chainlink BTC price feed integrated
- GMX V2 SDK loaded
- Wallet management

### Phase 2 — Data Pipeline
- 315,361 BTC 5-minute candles (3 years, Binance)
- 87 engineered technical features
- Saved as compressed Parquet

### Phase 3 — Model Development
- Volatility HMM (Calm/Normal/Turbulent) ← REAL EDGE
- Directional HMM (24h features) ← weak signal
- XGBoost classifier (TA only) ← no edge

### Week 1 — Funding Rate Pipeline
- Binance + Bybit 3-year funding history
- Cross-exchange spread features
- Causal forward-fill (no look-ahead)
- 11 new features added

### Week 2 — Network Flow Pipeline
- 5 daily metrics from blockchain.com (3 years)
- 1-day lag enforced (look-ahead protection)
- 22 new features added (120 total)

### Week 4 — XGBoost Retrain
- v1 baseline (TA only): 38.3% BUY precision
- v2 with alpha (120 cols): 36.8% BUY precision
- **Decisive negative result documented**

## Key Findings

1. **Direction is hard.** 1-hour BTC direction is not 
   predictable from TA + funding + flow features alone.
   
2. **Volatility is predictable.** Volatility regimes 
   cluster persistently (Markov property holds).

3. **Importance ≠ value.** Features can rank high in 
   tree model importance while not generalizing out-of-sample.

4. **Honest negative results matter.** 4 weeks saved 
   from deploying a non-edge model with real capital.

## Tech Stack

- Python 3.11
- pandas, numpy, scikit-learn
- XGBoost, hmmlearn
- Web3, GMX Python SDK
- Causal merge_asof patterns

## Next Steps

Pivoting from directional prediction to 
volatility-regime-based risk management.

## License

MIT - free to study and learn from.

## About

Built by [Your Name] - Python engineer focused on 
quantitative trading systems. Available for remote 
roles in Python/ML/quant development.

[Your LinkedIn] | [Your Email]
