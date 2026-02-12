# CLAUDE.md

## Project Overview

**acoes** is a quantitative finance repository containing algorithmic trading strategies for Brazilian equities. The algorithms are written in Python for the **Quantopian** backtesting platform and implement a value-investing approach combining fundamental metrics with momentum filtering.

**Author:** Gabriel (@drgabriellopes)
**Language:** Python (Quantopian API)
**Domain:** Algorithmic trading / quantitative finance — Brazilian stock market

## Repository Structure

```
acoes/
├── Ev ebit roepl mom6              # Base strategy: EV/EBIT + ROE/PL + 6-month momentum
├── Evebit roepl mom6 ibov          # Enhanced strategy with IBOV benchmark tracking
└── CLAUDE.md                       # This file
```

The repository is flat — all algorithm files live at the root level with no subdirectories. File names use Portuguese, space-separated descriptive naming reflecting the strategy parameters.

## Strategy Logic

Both algorithms implement the same core selection methodology:

1. **Universe:** Brazilian equities in the QTradableStocksUS universe
2. **Factor computation:**
   - `ev_to_ebit` — Enterprise Value / EBIT (lower is better value)
   - `roe_over_price_to_earnings` — ROE / PE ratio (lower percentile selected)
   - `momentum` — 6-month (126 trading days) price momentum
3. **Screening:** Filter to the bottom 10th percentile on both EV/EBIT and ROE/PE
4. **Selection:** Pick the top 20 stocks by momentum from the filtered set
5. **Portfolio rules:** Monthly rebalancing (month-end + 3 days offset), max leverage 1.0, max 33% per position

### Differences Between Versions

| Feature | `Ev ebit roepl mom6` | `Evebit roepl mom6 ibov` |
|---|---|---|
| Benchmark | None | IBOV (Ibovespa index) |
| Portfolio tracking | No | Tracks initial investment of 10,000 |
| Liquidity filter | None | AverageDollarVolume > 125,000 |
| Date handling | String format | `datetime` objects with `pytz.UTC` |
| Extra imports | — | `pytz`, `matplotlib`, `datetime`, `quandl` |

## Dependencies

These algorithms run on the Quantopian platform. There is no `requirements.txt` or local dependency file. Key libraries used:

- `quantopian.algorithm` — Core backtesting engine
- `quantopian.optimize` — Portfolio optimization (target weights, constraints)
- `quantopian.pipeline` — Data pipeline and factor research framework
- `numpy` / `pandas` — Numerical and data manipulation
- `pytz` — Timezone handling (v2 only)
- `matplotlib` — Visualization (v2 only)

## Development Workflow

### Running the Algorithms

These are **not standalone scripts**. They are designed to be copy-pasted into the Quantopian IDE or a compatible backtesting engine (e.g., Zipline). They cannot be run with a standard `python` command.

### No Build / Test / Lint Commands

The repository has no:
- Build system or scripts
- Test framework or test files
- Linter or formatter configuration
- CI/CD pipelines

Backtesting is performed directly on the Quantopian platform.

## Code Conventions

- **Custom factors** are defined as classes extending `quantopian.pipeline.factors.CustomFactor`
- **Entry point:** `initialize(context)` sets up the trading context and schedules
- **Rebalancing logic:** `rebalance(context, data)` builds and runs the pipeline, filters, and sets target weights
- **Window length convention:** 126 trading days = ~6 months
- **Percentile filtering:** `.rank(pct=True)` used for percentile-based screening
- **Naming:** File names are in Portuguese; code variables are in English

## Notes for AI Assistants

- File names contain spaces — always quote paths when using shell commands
- This is a Quantopian-specific codebase; the API is not standard Python and requires the Quantopian runtime
- The first file appears truncated in the repository (ends mid-expression at line 46)
- No `.gitignore` is present — be mindful when adding files
- There are no existing tests; any new code cannot be validated locally without a Quantopian-compatible engine
