# MORGOTH GMX

Autonomous BTC perpetuals trading bot for GMX V2 on Arbitrum.

## Vision
An ensemble ML-driven trading agent that:
- Reads BTC 5-min markets continuously
- Combines 5-10 independent prediction models
- Trades only on high-consensus signals (80%+ agreement)
- Uses Kelly criterion for position sizing
- Risk-managed with ATR-based stops
- Patient: most days takes no trades

## Architecture

### Layer 1: Data Pipeline
- Real-time BTC 5-min feed
- 40+ technical indicators
- Volume + order flow analysis
- Multi-timeframe (M5, M15, H1, H4)

### Layer 2: Prediction Models
- HMM: Regime detection (Bull/Bear/Ranging)
- GRU: 5-min direction prediction
- LSTM: 15-min trend confirmation
- XGBoost: Signal classification
- Divergence detector
- Order flow analyzer

### Layer 3: Decision Engine
- Voting system across all models
- Consensus threshold: 80%+
- Confidence scoring
- Below threshold = no action

### Layer 4: Risk & Execution
- Kelly criterion sizing (quarter Kelly)
- ATR-based stops
- Max 2% account risk per trade
- 3-10x leverage range
- GMX V2 SDK integration

### Layer 5: Monitoring
- Live dashboard (Streamlit)
- Telegram alerts
- Trade journal (SQLite)
- Performance metrics

## Roadmap

### Phase 1: Foundation (Week 1) ✓ STARTED
- Project structure
- Git/GitHub setup
- GMX connection from Algeria
- Data fetcher

### Phase 2: Data Pipeline (Week 1-2)
- Historical BTC data (3+ years)
- Feature engineering (40+ indicators)
- Real-time data feed

### Phase 3: Model Development (Week 2-3)
- Train HMM regime detector
- Train GRU/LSTM predictors
- Train XGBoost classifier
- Ensemble voting logic

### Phase 4: Execution Layer (Week 3)
- GMX V2 SDK integration
- Position management
- Risk management

### Phase 5: Dashboard & Alerts (Week 3-4)
- Streamlit dashboard
- Telegram notifications
- Logging system

### Phase 6: Live Testing (Week 4+)
- DRY_RUN mode first
- Micro positions ($0.50)
- Performance analysis
- Iterative improvement

## Status

Project initialized: June 9, 2026
Capital: $13.68 USDC.e (to be bridged to Arbitrum)
Stage: Foundation building

## Tech Stack

- Python 3.11+
- gmx_python_sdk
- pandas, numpy, scikit-learn
- TensorFlow/Keras (GRU/LSTM)
- XGBoost
- hmmlearn
- Streamlit (dashboard)
- python-telegram-bot
- SQLite (local DB)

## Safety

- Private keys in .env (gitignored)
- Maximum daily loss: 5% of balance
- Auto-pause on 15% drawdown
- All actions logged
- Telegram alerts on every trade

## License

Private during development.
Will be open-sourced after 3-6 months proven results.

---

Built with patience. Built with discipline. Built for the long game.
