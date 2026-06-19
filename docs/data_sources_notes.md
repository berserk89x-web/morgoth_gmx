# MORGOTH GMX — Data Sources Notes

Running log of alternative-data source research and decisions.

## Liquidation data — finding (2026-06-19)

**No free historical source as of June 2026.**

Probed candidates:
- **Binance** `fapi/v1/allForceOrders` → HTTP 400 "endpoint out of maintenance" (deprecated). `forceOrders` → auth-only (signed, API key).
- **Bybit** `v5/market/recent-trade` → works, but no liquidation flag in the trade structure. No historical liquidation REST.
- **OKX** `api/v5/public/liquidation-orders` → **works, free, no API key.** Returns real liquidation fields (`bkPx`, `posSide`, `side`, `sz`, `time`). **Recent only — no deep historical backfill.**
- **Hyperliquid** `info` API → no `liquidations` type (HTTP 422). Liquidation fills only appear per-address in `userFillsByTime` (not a global feed).

**OKX is the only free public endpoint — but forward-only.**

**Plan:** collect OKX liquidations forward in parallel, add as a feature in Week 8+ (live-only signal; cannot be merged into the existing 3-year historical training set). Paid backfill (e.g. Coinglass) deferred — not justified at current capital.

Research artifact `core/liquidation_fetcher.py` (Hyperliquid exploration) was intentionally **left uncommitted** — research only.

## Whale tracking — finding (2026-06-19)

Evaluated as the replacement Week 1. Outcome: **free whale APIs are paywall-limited.**

- **Glassnode** — API is Professional-only (no free API; free = web Studio).
- **CryptoQuant** — has exactly the right metrics (Exchange Whale Ratio, in/outflows, deep history) but API needs paid Professional/Premium.
- **Whale Alert** — free key exists but real-time only (~1h lookback, min ~$500k); historical is paid.
- **Bitquery** — free tier is a one-time 1,000-point trial, 10 rows/request — too throttled for bulk history.
- **Etherscan** — Ethereum only, does **not** cover BTC.
- **Blockchair / Blockchain.com / mempool.space** — free + historical, but **raw** large-tx data only (**no exchange attribution**).

So: no free, labeled, historical whale/exchange-flow API. Blockchair is viable for a DIY large-transaction proxy with history, but raw (no attribution).

**Decision: pivot Week 1 to funding rates** (full free historical across multiple exchanges, clean APIs, proven contrarian edge, better signal-to-effort). **Whale tracking parked** — revisit later as optional Blockchair "whale-lite" (raw large-tx counts/volume).

