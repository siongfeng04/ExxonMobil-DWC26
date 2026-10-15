# DWC26: ML-Assisted History Matching and 20-Year Production Forecast

ExxonMobil DataWorks Challenge 2026. The organisers simulated a reservoir 100 times with different values of four
uncertain parameters, and provided the real field's production history for 1998–2008. We find the parameter values
that reproduce that history (**history matching**) and forecast oil, gas and water for the **10 best matches** to 2028.

**Approach:** train a fast ML proxy of the simulator → search it for parameters that match the real field → forecast
the 10 best matches → fill the organiser's template.

## Results

|                                                      |                                                                                |
| ---------------------------------------------------- | ------------------------------------------------------------------------------ |
| Best history match                                   | misfit **0.442**, better than every provided simulation (best: case 76, 0.453) |
| Proxy accuracy on unseen simulations                 | < 0.1 % error on 2008 cumulatives                                              |
| First forecast quarter vs unseen real data (2008-Q1) | oil **+1.2 %**, water **−0.3 %**                                               |

Forecast cumulatives at 2028-01-01 across the 10 delivered cases:

|               | P10   | P50   | P90   |
| ------------- | ----- | ----- | ----- |
| Oil [MMSTB]   | 55.94 | 55.95 | 56.03 |
| Gas [MMMSCF]  | 20.32 | 20.32 | 20.35 |
| Water [MMSTB] | 86.58 | 87.12 | 87.77 |

Deliverable: [outputs/result/09 Template Deliverable.xlsx](outputs/result/09%20Template%20Deliverable.xlsx)

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

| Notebook                                                                          | What it does                                                    |
| --------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| [00_DWC2026.ipynb](notebook/00_DWC2026.ipynb)                                     | Cleans the raw files in `dwc26/` into `Reservoir_ML_Data/`; the curve files used later are copied to `data/` |
| [01_Proxy_Model.ipynb](notebook/01_Proxy_Model.ipynb)                             | Compares proxy models and forecast methods                      |
| [02_History_Matching_Forecast.ipynb](notebook/02_History_Matching_Forecast.ipynb) | History matching, optimiser comparison, validation, deliverable |

The full analysis, charts and reasoning are in the notebooks. [pipeline/](pipeline/) holds the same logic as notebook 02, without charts.

## Key decisions

| Decision                                    | Chosen over                 | Why                                                                                           |
| ------------------------------------------- | --------------------------- | --------------------------------------------------------------------------------------------- |
| Inputs = the 4 case parameters              | Grid-block data             | Grid data doesn't exist for new parameter sets, and it is determined by the parameters anyway |
| PCA + **Gaussian Process** proxy            | Ridge, neural network (MLP) | Most accurate on validation cases by a wide margin, and suits small data (70 training cases)  |
| **Decline-curve** forecast                  | GRU neural network, blend   | Lowest backtest error; physically plausible                                                   |
| **Sobol + L-BFGS-B** optimiser              | Differential Evolution, ES-MDA | All give the same best match and the same forecast; Sobol finds the best match exactly on every seed. ES-MDA is 15× cheaper, the choice if the real simulator were used |
| 10 cases = **best matches first, no near-copies** | Most spread-out acceptable matches; plain top 10 | FAQ Q8–9: selection driven by history-match quality. All 10 are within 2 % of the best misfit; the plain top 10 would be one point repeated |
| Candidates from the **full parameter range**  | Only the 100 provided cases, or one subset | FAQ Q10: the search spans all 100 cases (train, validation, test); the best provided case (76) is beaten |
| Forecast **anchored** to last observed rate | Unanchored                  | Best oil error in the field backtest                                                          |

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

**Quick run (uses the cleaned data already in the repo)**

1. Put the organiser's starter kit in `dwc26/` (gitignored). Only the official template is read from there.
2. Run `python -m pipeline.main` from the repo root, or open notebook 02 and run all cells (~6 minutes).

The cleaned inputs are committed in `data/`, so the cleaning step is not needed for this.

**Full run from the raw files (optional, to reproduce `data/`)**

1. Put the starter kit in `dwc26/` as above.
2. Run [notebook 00](notebook/00_DWC2026.ipynb). It reads `dwc26/TestCases/` and writes the cleaned files to
   `Reservoir_ML_Data/`. The grid-cleaning part (Sections 2–4) is slow (~1.6 GB of Excel)
   and its outputs are not used later; to skip it, run only the setup cell and Sections 5–6 (production curves).
3. Copy `Simulation_Rates_Consolidated.xlsx` and `Simulation_Cumulatives_Consolidated.xlsx` from `Reservoir_ML_Data/`,
   and `01 Introduction.xlsx` from `dwc26/TestCases/`, into `data/`.
4. Run the pipeline or notebook 02 as in the quick run.

## Repository layout

```
data/                 prepared inputs: parameters, simulated curves, observed history
dwc26/                organiser starter kit (not in git)
Reservoir_ML_Data/    cleaned outputs of notebook 00 (not in git)
notebook/             00 data prep, 01 proxy model, 02 history matching + forecast
pipeline/             config, data loading, proxy, history matching, forecast, end-to-end run
outputs/
├── result/           Excel outputs
│   ├── 09 Template Deliverable.xlsx      the submission (10 cases, 2008 → 2028), Power BI source
│   ├── history_match_results.xlsx        selected cases, acceptable matches, blind test, field backtest
│   └── Forecast_Deliverable_DEMO.xlsx    notebook 01's placeholder demo (not the submission)
├── figures/          one folder per notebook, charts numbered in the order they appear
│   ├── 01_Proxy_Model/                   01–06: proxy accuracy, sensitivity, forecast backtests
│   ├── 02_History_Matching_Forecast/     01–14: every chart in notebook 02 (see below)
│   └── pipeline/                         01–03: key charts saved by python -m pipeline.main
└── models/           notebook 01's saved models (not in git)
```

Notebook 02 charts, for the slides:

| File | Shows |
|---|---|
| `01_date_shift_check` | Observed vs simulated rates before/after the one-quarter date shift |
| `02_proxy_accuracy_test_cases` | Proxy vs simulator on the 15 unseen test cases |
| `03_provided_cases_misfit` | The 15 provided simulations closest to the real field |
| `04_misfit_landscape` | Where in the parameter space the good matches are |
| `05_history_match_curves` | 1998–2008: provided curves, the 10 matched cases, the real field |
| `06_matched_parameters` | What the history pins down (porosity) and what it doesn't (fault) |
| `07_optimizer_convergence` | Sobol + L-BFGS-B vs DE vs ES-MDA: best misfit vs proxy calls |
| `08_optimizer_search_coverage` | Where each optimizer spent its evaluations |
| `09_optimizer_forecast_2028` | 2028 forecast from each optimizer's 10 cases |
| `10_blind_test` | Recovered vs true parameters on cases with a known answer |
| `11_field_backtest_errors` | Forecast error on real held-out quarters, by method |
| `12_backtest_2006_detail` | Forecast from 2006-07 vs what the field really did |
| `13_forecast_rates` | Oil and water rate forecast to 2028, with the unseen 2008-Q1 point |
| `14_forecast_cumulatives` | 20-year cumulative forecast of the 10 delivered cases |
