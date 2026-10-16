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
    D --> E[Pick 10 best matches,<br/>no near-copies]
    E --> F[Forecast to 2028]
    F --> G[Template deliverable]
```

| Notebook                                                                          | What it does                                                    | Run time |
| --------------------------------------------------------------------------------- | --------------------------------------------------------------- | -------- |
| [00_Data_Preparation.ipynb](notebook/00_Data_Preparation.ipynb)                   | Cleans the raw files in `dwc26/` into `data/`                   | ~40 min  |
| [01_Proxy_Model.ipynb](notebook/01_Proxy_Model.ipynb)                             | Compares proxy models and forecast methods                      | ~12 min  |
| [02_History_Matching_Forecast.ipynb](notebook/02_History_Matching_Forecast.ipynb) | History matching, optimiser comparison, validation, deliverable | ~6 min   |

The full analysis, charts and reasoning are in the notebooks; every chart has a short "What it shows" and "Findings"
note under it. [pipeline/](pipeline/) holds the same logic as notebook 02, without charts.

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

Requires Python 3.11 (tested with 3.11.9).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

`torch` is only needed for notebook 01 (the GRU comparison); a CPU-only install is enough.
The notebooks can be opened from the repo root or from inside `notebook/`; they find the repo root themselves.

**Quick run (uses the cleaned data already in `data/`)**

Run `python -m pipeline.main` from the repo root (~3 minutes), or open notebook 02 and run all cells (~6 minutes).
Both write the deliverable to `outputs/result/`. The case parameters (`01 Introduction.xlsx`) and the official
template are read from `dwc26/TestCases/`.

The results are reproducible: seeds are fixed, and re-running the pipeline gives exactly the submitted deliverable
and `history_match_results.xlsx`.

**Full run from the raw files (optional, to rebuild `data/`)**

1. Run [notebook 00](notebook/00_Data_Preparation.ipynb) (~40 minutes). It reads `dwc26/TestCases/` and writes the
   cleaned files to `data/`. It needs the six grid files `03`–`08` (Train / Validation / Test Cases), which are not
   included: copy them from the organiser's starter kit into `dwc26/TestCases/` (see [Large files](#large-files)).
   The grid-cleaning part (Sections 2–4) takes most of the time (~1.6 GB of Excel)
   and its outputs are not used later; to skip it, run only the setup cell and Sections 5–7 (case parameters and
   production curves).
2. Optionally run [notebook 01](notebook/01_Proxy_Model.ipynb) to repeat the proxy and forecast-method comparison.
3. Run the pipeline or notebook 02 as in the quick run.

## Repository layout

```
data/                 cleaned outputs of notebook 00
├── Simulation_Rates_Consolidated.xlsx        simulated rates of the 100 cases + observed history (used later)
├── Simulation_Cumulatives_Consolidated.xlsx  simulated cumulatives of the 100 cases + observed history (used later)
├── Cleaned (Overall Data)/                   grid data, all blocks (Parquet; not used later; not included)
└── Usable Data/                              grid data, active blocks only (Parquet; not used later; not included)
dwc26/TestCases/      organiser starter kit: raw files, case parameters, official template
                      (only files 01, 02 and 09 are included; grid files 03–08 are not)
notebook/             00 data preparation, 01 proxy model, 02 history matching + forecast
pipeline/             config, data loading, proxy, history matching, forecast, end-to-end run
requirements.txt      Python packages (tested versions listed at the top)
outputs/
├── result/           Excel outputs
│   ├── 09 Template Deliverable.xlsx      the submission (10 cases, 2008 → 2028)
│   └── history_match_results.xlsx        selected cases, acceptable matches, blind test, field backtest
├── figures/          one folder per notebook, charts numbered in the order they appear
│   ├── 01_Proxy_Model/                   01–06: every chart in notebook 01 (see below)
│   ├── 02_History_Matching_Forecast/     01–14: every chart in notebook 02 (see below)
│   └── pipeline/                         01–03: key charts saved by python -m pipeline.main
└── models/           notebook 01's saved models (created when notebook 01 is run)
```

### Large files

Some large files are only needed to rebuild the grid data in notebook 00, so they are not included in the
submission or on GitHub:

| File | Size | How to get it |
|---|---|---|
| `dwc26/TestCases/03`–`08` (Train / Validation / Test Cases) | 166–331 MB each | Organiser's starter kit |
| `data/Cleaned (Overall Data)/`, `data/Usable Data/` | ~580 MB in total | Run notebook 00 (needs the files above) |

Everything the quick run needs is included: `data/Simulation_*_Consolidated.xlsx` and `dwc26/TestCases/01`, `02`
and `09`.

Notebook 01 charts:

| File | Shows |
|---|---|
| `01_history_curves_by_split` | Simulated 1998–2008 cumulatives of all 100 cases, by train / validation / test split |
| `02_history_proxy_parity_test` | Proxy vs simulator 2008 cumulatives on the 15 unseen test cases |
| `03_history_proxy_examples_test` | Proxy vs simulator oil and water curves for three test cases |
| `04_sensitivity_tornado` | How much each parameter moves 2008 oil and water (porosity dominates) |
| `05_backtest_examples_test` | Decline curves vs GRU vs blend, forecasting 2006–2008 from data up to 2006 |
| `06_forecast_2028_val_test` | Decline-curve vs GRU forecast to 2028 for the validation and test cases |

Notebook 02 charts:

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
