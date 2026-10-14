# DWC26: ML-Assisted History Matching and 20-Year Production Forecast

ExxonMobil DataWorks Challenge 2026. The organisers simulated a reservoir 100 times with different values of four
uncertain parameters, and provided the real field's production history for 1998–2008. We find the parameter values
that reproduce that history (**history matching**) and forecast oil, gas and water for the **10 best matches** to 2028.

**Approach:** train a fast ML proxy of the simulator → search it for parameters that match the real field → forecast
the 10 best matches → fill the organiser's template.

## Results

| | |
|---|---|
| Best history match | misfit **0.442**, better than every provided simulation (best: case 76, 0.453) |
| Proxy accuracy on unseen simulations | < 0.1 % error on 2008 cumulatives |
| First forecast quarter vs unseen real data (2008-Q1) | oil **+1.3 %**, water **−0.3 %** |

Forecast cumulatives at 2028-01-01 across the 10 delivered cases:

| | P10 | P50 | P90 |
|---|---|---|---|
| Oil [MMSTB] | 55.94 | 56.20 | 56.44 |
| Gas [MMMSCF] | 20.32 | 20.42 | 20.50 |
| Water [MMSTB] | 86.22 | 87.81 | 89.29 |

Deliverable: [outputs/09 Template Deliverable.xlsx](outputs/09%20Template%20Deliverable.xlsx)

## Workflow

```mermaid
flowchart LR
    A[Organiser Excel files] --> B[00 Data preparation]
    B --> C[01 Proxy model<br/>4 params → curves]
    C --> D[02 History matching<br/>Sobol + L-BFGS-B]
    D --> E[Pick 10 diverse matches]
    E --> F[Forecast to 2028]
    F --> G[Template deliverable]
```

| Notebook | What it does |
|---|---|
| [00_DWC2026.ipynb](notebook/00_DWC2026.ipynb) | Cleans the raw files and builds the curve files in `data/` |
| [01_Proxy_Model.ipynb](notebook/01_Proxy_Model.ipynb) | Compares proxy models and forecast methods |
| [02_History_Matching_Forecast.ipynb](notebook/02_History_Matching_Forecast.ipynb) | History matching, optimiser comparison, validation, deliverable |

The full analysis, charts and reasoning are in the notebooks. [pipeline/](pipeline/) holds the same logic as notebook 02, without charts.

## Key decisions

| Decision | Chosen over | Why |
|---|---|---|
| Inputs = the 4 case parameters | Grid-block data | Grid data doesn't exist for new parameter sets, and it is determined by the parameters anyway |
| PCA + **Gaussian Process** proxy | Ridge, neural network (MLP) | Most accurate on validation cases by a wide margin, and suits small data (70 training cases) |
| **Decline-curve** forecast | GRU neural network, blend | Lowest backtest error; physically plausible |
| **Sobol + L-BFGS-B** optimiser | Differential Evolution | Both find the same best match; Sobol covers the acceptable region more evenly |
| 10 cases = spread-out acceptable matches | Top 10 by misfit | Shows the real parameter uncertainty instead of ten near-copies |
| Forecast **anchored** to last observed rate | Unanchored | Best oil error in the field backtest |

## Key findings

- **Porosity drives everything** and is the one parameter the history pins down (≈ 1.38).
- **Fault transmissibility is unconstrained** by the history and barely affects the forecast.
- **The model can't reproduce the field's late-time water** (~60 % water cut vs ~54 % at best), so the forecast is
  anchored to the observed rates.
- **The forecast range is narrow** because it covers parameter uncertainty only. The bigger uncertainties are the
  water-cut gap and the future well schedule, and there is no data after 2008 to check the 20-year horizon.

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

1. Put the organiser's starter kit in `dwc26/` (gitignored). Only the official template is read from there.
2. Run `python -m pipeline.main` from the repo root, or open notebook 02 and run all cells (~6 minutes).

**Known issue:** notebook 01 still reads the template from the repo root, where it has been removed, and expects the
repo root as its working directory.

## Repository layout

```
data/         prepared inputs: parameters, simulated curves, observed history
dwc26/        organiser starter kit (not in git)
notebook/     00 data prep, 01 proxy model, 02 history matching + forecast
pipeline/     config, data loading, proxy, history matching, forecast, end-to-end run
outputs/      deliverable, history_match_results.xlsx, figures/
```

> The datasets are provided exclusively for the DataWorks Challenge 2026. Do not publish them externally.
