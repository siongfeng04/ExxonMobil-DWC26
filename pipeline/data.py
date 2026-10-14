"""Load the prepared Excel files in `data/` into case-ordered numpy cubes.

The two `Simulation_*_Consolidated.xlsx` files are `02 Field Production.xlsx` reshaped by `notebook/DWC2026.ipynb`
into long format (Date, Case, oil, gas, water): one block of rows per case, values identical to the original sheets.
The consolidation lost one column name: the observed field history (`RAMP_History` in the original) comes through
as `UNKNOWN_COL_101`. It is the history-matching target, not junk.

Two conventions matter for history matching:

- Simulated rates (CASE 1 ... CASE 100) are averages over the period ENDING at the timestamp.
- Observed rates are averages over the period STARTING at the timestamp. E.g. the observed 15,507.77 STB/d dated
  1998-10-01 is the simulators' 15,507.77 dated 1999-01-01. Cumulatives agree in both, so observed rates are
  shifted one quarter later to compare against the proxy.
"""
import re

import numpy as np
import pandas as pd

from .config import DATA_DIR, PARAM_COLS

INTRO_FILE = DATA_DIR / '01 Introduction.xlsx'
CURVE_FILES = {  # cube axis order is always [oil, gas, water]
    'rate': (DATA_DIR / 'Simulation_Rates_Consolidated.xlsx',
             ['Oil_production_rate_STB_d', 'Gas_production_rate_MSCF_d', 'Water_production_rate_STB_d']),
    'cum': (DATA_DIR / 'Simulation_Cumulatives_Consolidated.xlsx',
            ['Oil_production_cumulative_STB', 'Gas_production_cumulative_MSCF', 'Water_production_cumulative_STB']),
}
OBSERVED = 'UNKNOWN_COL_101'  # observed field history (RAMP_History), unnamed by the consolidation step


def case_number(label):
    return int(re.search(r'\d+', str(label)).group())


def load_params():
    """Uncertainty table: (100, 4) DataFrame indexed by case number."""
    p = pd.read_excel(INTRO_FILE, sheet_name='Uncertainty Values')
    p.index = p['Item'].map(case_number)
    p.index.name = 'case'
    return p[PARAM_COLS].sort_index()


def _read_curves(path, cols):
    """(100, n_dates, 3) simulated cube, (n_dates, 3) observed curves and the dates, from one consolidated file."""
    df = pd.read_excel(path)
    df['Date'] = pd.to_datetime(df['Date'])
    obs = df[df['Case'] == OBSERVED].set_index('Date').sort_index()
    sim = df[df['Case'].str.startswith('CASE')].copy()
    sim['case'] = sim['Case'].map(case_number)
    cube = np.stack([sim.pivot(index='case', columns='Date', values=c).sort_index().to_numpy(float) for c in cols],
                    axis=-1)
    assert sorted(sim['case'].unique()) == list(range(1, 101))
    assert not np.isnan(cube).any(), f'missing simulated values in {path.name}'
    return cube, obs[cols].to_numpy(float), pd.DatetimeIndex(obs.index)


def load_field():
    """Simulated curves for all 100 cases and the observed history, at the 41 quarterly dates 1998-01 → 2008-01.

    Returns dict with
      dates        DatetimeIndex (41,)
      rate, cum    (100, 41, 3) simulated [oil, gas, water], rates in simulator convention
      obs_rate     (41, 3) observed rates shifted to simulator convention (rate over the quarter ending at date)
      obs_cum      (41, 3) observed cumulatives
      obs_next     (3,)  observed rate for the quarter 2008-01 → 2008-04, i.e. simulator-convention rate at
                   2008-04-01: one quarter of real data beyond the simulations, used to check the forecast.
    """
    rate, obs_fwd, dates = _read_curves(*CURVE_FILES['rate'])
    cum, obs_cum, cum_dates = _read_curves(*CURVE_FILES['cum'])
    assert dates.equals(cum_dates) and len(dates) == rate.shape[1]

    obs_rate = np.vstack([np.zeros((1, 3)), obs_fwd[:-1]])
    return {'dates': dates, 'rate': rate, 'cum': cum, 'obs_rate': obs_rate, 'obs_cum': obs_cum,
            'obs_next': obs_fwd[-1]}
