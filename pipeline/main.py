"""DWC26 end-to-end pipeline: ML proxy → history matching (optimisation) → 20-year forecast → filled template.

Run from the repository root:   python -m pipeline.main

Steps
1. Blind test on cases 86–100: the proxy and the history matcher never see these cases. Each case's simulated
   history is treated as "observed"; we check the matcher recovers its parameters and its forecast.
2. Field backtest: history-match the real observations up to a cutoff, forecast the held-out quarters, compare
   with what the field actually did. This chooses between forecast variants on real data.
3. Final run: proxy on all 100 cases → history match to 2008 → forecast to 2028 for the 10 selected cases →
   outputs/result/09 Template Deliverable.xlsx, outputs/result/history_match_results.xlsx and
   outputs/figures/pipeline/01-03.
"""
import sys
import time
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from .config import (COLORS, DELIVERABLE_PATH, FC_DATES, FIG_DIR, N_DELIVER, OUT_DIR, PARAM_COLS, RESULT_DIR,
                     RESULTS_PATH, TEMPLATE_PATH, TEST_CASES, TRAIN_CASES, UNITS, VARS)
from .data import load_field, load_params
from .forecast import forecast, write_template
from .history_match import Misfit, ParamBox, history_match
from .proxy import CurveProxy, cum_correction, decline_forecast

warnings.filterwarnings('ignore', category=ConvergenceWarning)
ix = lambda cases: np.asarray(cases) - 1
pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 20)


def blind_test(field, params, corr):
    """History-match each test case's own simulated history with a proxy trained on cases 1–85."""
    known = [c for c in params.index if c not in TEST_CASES]
    proxy = CurveProxy().fit(params.loc[known], field['rate'][ix(known)], field['cum'][ix(known)])
    box, span = ParamBox(params), (params.max() - params.min()).to_numpy()
    rows = []
    for c in TEST_CASES:
        rate, cum = field['rate'][c - 1], field['cum'][c - 1]
        best, _ = history_match(Misfit(proxy, rate, cum), box, n_sobol_log2=11, n_starts=5, n_keep=1)
        both = np.vstack([best[PARAM_COLS].to_numpy(), params.loc[[c]].to_numpy()])  # [recovered, true]
        fc = forecast(proxy, both, field['dates'], rate, cum, corr)['cum'][:, -1]
        rows.append({'case': c, 'misfit': best['misfit'][0],
                     **{f'true {p}': both[1, k] for k, p in enumerate(PARAM_COLS)},
                     **{f'recovered {p}': both[0, k] for k, p in enumerate(PARAM_COLS)},
                     **{f'{p} error % of range': 100 * abs(both[0, k] - both[1, k]) / span[k]
                        for k, p in enumerate(PARAM_COLS)},
                     **{f'{v} cum 2028 diff %': 100 * (fc[0, i] - fc[1, i]) / fc[1, i] for i, v in enumerate(VARS)}})
    return pd.DataFrame(rows).set_index('case')


def field_backtest(field, params, corr, cutoffs=('2005-01-01', '2006-01-01', '2006-07-01')):
    """Forecast variants judged against the real observed quarters after each cutoff."""
    dates, o, oc = field['dates'], field['obs_rate'], field['obs_cum']
    rows = []
    for cut in cutoffs:
        c = dates.get_loc(pd.Timestamp(cut))
        h, bt = slice(0, c + 1), dates[c + 1:]
        proxy = CurveProxy().fit(params, field['rate'][:, h], field['cum'][:, h])
        sel, _ = history_match(Misfit(proxy, o[h], oc[h]), ParamBox(params), n_sobol_log2=12, n_starts=10)
        theta = sel[PARAM_COLS].to_numpy()
        variants = {}
        for name, anchor in [('A: matched model', False), ('A+: matched model, anchored', True)]:
            fc = forecast(proxy, theta, dates[h], o[h], oc[h], corr, fc_dates=bt, anchor=anchor)
            cum = fc['cum'] if anchor else fc['cum'] - fc['cum_hist'][:, -1:] + oc[c]  # same cum start for both
            variants[name] = (fc['rate'], cum)
        r_obs, c_obs, _ = decline_forecast(o[None, h], oc[None, h], dates[h], bt, 5, corr)
        variants['D: decline on observed only'] = (r_obs, c_obs)
        for name, (r, cm) in variants.items():
            row = {'cutoff': cut, 'quarters ahead': len(bt), 'method': name}
            for v in (0, 2):
                row[f'{VARS[v]} rate MAPE %'] = 100 * np.mean(np.abs(np.median(r[:, :, v], 0) - o[c + 1:, v]) / o[c + 1:, v])
                row[f'{VARS[v]} Δcum error %'] = 100 * abs(np.median(cm[:, -1, v]) - oc[-1, v]) / (oc[-1, v] - oc[c, v])
            rows.append(row)
    return pd.DataFrame(rows)


def plot_history_match(field, sel_hist, fname):
    dates = field['dates']
    panels = [('rate', 0, 'Oil rate', 'STB/d'), ('rate', 2, 'Water rate', 'STB/d'),
              ('cum', 0, 'Oil cumulative', 'MMSTB'), ('cum', 2, 'Water cumulative', 'MMSTB')]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
    for ax, (kind, v, title, unit) in zip(axes.flat, panels):
        s = 1e6 if kind == 'cum' else 1
        obs = field['obs_rate'] if kind == 'rate' else field['obs_cum']
        for j, curve in enumerate(field[kind][:, :, v]):
            ax.plot(dates, curve / s, color=COLORS['provided'], lw=0.7, alpha=0.7,
                    label='Provided simulations (100 cases)' if j == 0 else None)
        for j, curve in enumerate(sel_hist[kind][:, :, v]):
            ax.plot(dates, curve / s, color=COLORS['matched'], lw=1.6,
                    label=f'History-matched proxy ({len(sel_hist[kind])} cases)' if j == 0 else None)
        ax.scatter(dates, obs[:, v] / s, s=14, color=COLORS['actual'], zorder=3, label='Observed field')
        ax.set_title(title)
        ax.set_ylabel(unit)
    axes[0, 0].legend(loc='lower center')
    fig.suptitle('Provided vs. history-matched curves, 1998–2008', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_forecast(field, fc, fname):
    dates = field['dates']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, v in zip(axes, [0, 1, 2]):
        for j, curve in enumerate(field['cum'][:, :, v]):
            ax.plot(dates, curve / 1e6, color=COLORS['provided'], lw=0.6, alpha=0.6,
                    label='Provided simulations' if j == 0 else None)
        for j, curve in enumerate(fc['cum'][:, :, v]):
            ax.plot(FC_DATES, curve / 1e6, color=COLORS['matched'], lw=1.2 if j else 2,
                    alpha=0.55 if j else 1, label='Case 1 (best match)' if j == 0 else ('Cases 2–10' if j == 1 else None))
        ax.scatter(dates, field['obs_cum'][:, v] / 1e6, s=10, color=COLORS['actual'], zorder=3, label='Observed field')
        ax.axvline(dates[-1], color=COLORS['muted'], lw=1, ls=':')
        ax.annotate(f'{fc["cum"][0, -1, v] / 1e6:.1f}', (FC_DATES[-1], fc['cum'][0, -1, v] / 1e6), xytext=(4, 0),
                    textcoords='offset points', va='center', fontsize=8.5, color='#52514e')
        ax.set_title(f'{VARS[v].title()} production cumulative')
        ax.set_ylabel(f'MM{UNITS[v]}')
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.margins(x=0.08)
    axes[0].legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_parameters(params, accepted, selected, fname):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    rng = np.random.default_rng(0)
    groups = [('Provided cases', params, COLORS['provided'], 18), ('Acceptable matches', accepted, '#9fc2ee', 10),
              ('Selected 10', selected, COLORS['matched'], 40)]
    for ax, p in zip(axes, PARAM_COLS):
        for y, (_, df, color, size) in enumerate(groups):
            ax.scatter(df[p], y + rng.uniform(-0.15, 0.15, len(df)), s=size, color=color, edgecolor='#fcfcfb',
                       linewidth=0.8)
        ax.set_yticks(range(len(groups)), [g[0] for g in groups] if p == PARAM_COLS[0] else [])
        ax.set_title(p)
        ax.grid(axis='y', visible=False)
        if p == 'Permeability Multiplier':
            ax.set_xscale('log')
            ax.set_xticks([0.5, 1, 2, 5, 10], ['0.5', '1', '2', '5', '10'])
    fig.suptitle('What the production history constrains', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    t0 = time.time()
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore[attr-defined]  # Windows consoles default to cp1252
    plt.rcParams.update({
        'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'axes.edgecolor': '#c3c2b7',
        'axes.grid': True, 'grid.color': '#e1e0d9', 'grid.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelcolor': '#52514e',
        'xtick.color': '#898781', 'ytick.color': '#898781', 'axes.titlesize': 11, 'axes.titleweight': 'bold',
        'legend.frameon': False, 'font.size': 9.5,
    })
    for d in (FIG_DIR, RESULT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    field, params = load_field(), load_params()
    dates = field['dates']
    corr = cum_correction(field['rate'][ix(TRAIN_CASES)], field['cum'][ix(TRAIN_CASES)], dates)
    print(f'Loaded 100 cases × {len(dates)} quarters ({dates[0].date()} → {dates[-1].date()}); '
          f'cum correction {dict(zip(VARS, corr.round(4)))}')

    print('\n[1/3] Blind history-matching test on cases 86–100 (proxy trained on 1–85)')
    blind = blind_test(field, params, corr)
    print(blind.filter(regex='misfit|%').describe().loc[['mean', 'max']].round(2).T)

    print('\n[2/3] Field backtest: forecast held-out observed quarters')
    backtest = field_backtest(field, params, corr)
    print(backtest.round(2).to_string(index=False))
    print(backtest.groupby('method', sort=False).mean(numeric_only=True).drop(columns='quarters ahead').round(2))

    print('\n[3/3] Final history match on the observed field (proxy trained on all 100 cases)')
    proxy = CurveProxy().fit(params, field['rate'], field['cum'])
    misfit = Misfit(proxy, field['obs_rate'], field['obs_cum'])
    selected, accepted = history_match(misfit, ParamBox(params), n_keep=N_DELIVER)
    provided = pd.Series(misfit.from_curves(field['rate'], field['cum']), index=params.index)
    print(f'Best provided simulation: case {provided.idxmin()} (misfit {provided.min():.3f}); '
          f'best match found: {selected["misfit"][0]:.3f}; acceptable matches: {len(accepted)}')

    fc = forecast(proxy, selected[PARAM_COLS].to_numpy(), dates, field['obs_rate'], field['obs_cum'], corr)
    raw = forecast(proxy, selected[PARAM_COLS].to_numpy(), dates, field['obs_rate'], field['obs_cum'], corr,
                   anchor=False)
    print('Out-of-sample check, first forecast quarter (2008-01 → 2008-04) vs the observed rate:')
    for v in (0, 2):
        actual = field['obs_next'][v]
        print(f'  {VARS[v]:5s} observed {actual:7.0f} | anchored forecast {np.median(fc["rate"][:, 0, v]):7.0f} '
              f'({100 * (np.median(fc["rate"][:, 0, v]) / actual - 1):+.1f}%) | unanchored '
              f'{np.median(raw["rate"][:, 0, v]):7.0f} ({100 * (np.median(raw["rate"][:, 0, v]) / actual - 1):+.1f}%)')

    write_template(fc['cum'], TEMPLATE_PATH, DELIVERABLE_PATH)
    summary = selected.copy()
    summary.index = [f'Case {k + 1}' for k in range(len(summary))]
    for v, (name, unit) in enumerate(zip(VARS, UNITS)):
        summary[f'{name} cum 2008 [{unit}]'] = field['obs_cum'][-1, v]
        summary[f'{name} cum 2028 [{unit}]'] = fc['cum'][:, -1, v]
    with pd.ExcelWriter(RESULTS_PATH) as xw:
        summary.to_excel(xw, sheet_name='Selected cases')
        accepted.to_excel(xw, sheet_name='Acceptable matches', index=False)
        provided.rename('misfit').sort_values().to_excel(xw, sheet_name='Provided case misfit')
        blind.to_excel(xw, sheet_name='Blind test')
        backtest.to_excel(xw, sheet_name='Field backtest', index=False)
    print('\nSelected cases written to the template:')
    print(summary.round(3).to_string())

    plot_history_match(field, {'rate': fc['rate_hist'], 'cum': fc['cum_hist']}, '01_history_match.png')
    plot_forecast(field, fc, '02_forecast_2028.png')
    plot_parameters(params, accepted, selected, '03_matched_parameters.png')
    root = OUT_DIR.parent
    print(f'\nWrote {DELIVERABLE_PATH.relative_to(root)}, {RESULTS_PATH.relative_to(root)} and '
          f'{FIG_DIR.relative_to(root)}/01-03 in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
