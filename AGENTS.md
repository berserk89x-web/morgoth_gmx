# MORGOTH GMX - AGENT RULES

## Project Identity
MORGOTH GMX is an autonomous BTC trading bot.
Long-term project. Quality over speed.

## Coding Standards
- Python 3.11+ syntax
- Type hints required
- Docstrings on all functions
- Black formatter for code style
- pytest for tests

## Trading Rules (NEVER violate)
1. NEVER trade without ensemble consensus >= 80%
2. NEVER risk more than 2% per trade
3. NEVER trade in DRY_RUN mode marked as live
4. ALWAYS use ATR-based stops
5. ALWAYS log every decision
6. ALWAYS send Telegram alert on action

## Development Rules
1. Test in DRY_RUN before any live trade
2. Backtest every strategy on historical data
3. One feature at a time (no scope creep)
4. Commit progress daily
5. Document every model change

## File Organization
- core/ = Business logic
- execution/ = GMX interaction
- models/ = ML models
- data/ = Data pipeline
- dashboard/ = Streamlit UI
- backtest/ = Historical testing
- alerts/ = Notifications
- config/ = Settings
- tests/ = pytest tests

## Security
- NEVER commit .env to git
- NEVER log private keys
- NEVER share keys in chat
- Use environment variables for secrets

## Performance Targets
- Phase 1 (Month 1): Foundation working
- Phase 2 (Month 2): First DRY_RUN signals
- Phase 3 (Month 3): First live $0.50 trades
- Phase 4 (Month 6): Proven edge or pivot
- Phase 5 (Month 12): $500+ portfolio

## When Confused
- Refer to README.md for vision
- Refer to docs/ for architecture
- Don't add features not in roadmap
- Ask user before major changes

---

This file guides Claude Code through the MORGOTH GMX build.
