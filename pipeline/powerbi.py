"""Data for the Power BI dashboard, plus preview charts of what the dashboard pages will show.

Run from the repository root:   python -m pipeline.powerbi

Writes
- outputs/result/case_history_forecast.xlsx: one sheet per delivered case (observed history 1998–2008, the case's
  matched-model history and its forecast to 2028, with the case's parameters and misfit on every row), an
  'All Cases' sheet stacking the 10 case sheets, a 'Parameters' sheet and a 'Read Me' sheet.
- outputs/result/history_matching.xlsx: the 100 provided simulation curves vs the observed field vs the selected
  cases' matched curves, and every scored parameter set (provided, acceptable, selected) with its misfit split into
  the four data blocks.
- outputs/result/optimizer_comparison.xlsx: Sobol + L-BFGS-B vs Differential Evolution vs ES-MDA (3 seeds each),
  as in notebook/02_History_Matching_Forecast section 7b: best misfit, cost, convergence, selected cases, and the
  misfit of the 100 provided simulations as the baseline.
- outputs/figures/powerbi/: preview charts built from exactly the rows in those workbooks.
"""
import sys
import time
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl.utils import get_column_letter
from sklearn.exceptions import ConvergenceWarning

from .config import COLORS, FC_DATES, N_DELIVER, OUT_DIR, PARAM_COLS, RESULT_DIR, SEED, TRAIN_CASES, UNITS
from .data import load_field, load_params
from .forecast import forecast
from .history_match import ACCEPT_TOL, Misfit, ParamBox, history_match, history_match_de, history_match_esmda
from .proxy import CurveProxy, cum_correction

warnings.filterwarnings('ignore', category=ConvergenceWarning)

FIG_DIR = OUT_DIR / 'figures' / 'powerbi'
CASES_PATH = RESULT_DIR / 'case_history_forecast.xlsx'
OPT_PATH = RESULT_DIR / 'optimizer_comparison.xlsx'
HM_PATH = RESULT_DIR / 'history_matching.xlsx'

PHASES = ['Oil', 'Gas', 'Water']  # cube axis order [oil, gas, water]
BLOCKS = ['Oil Rate', 'Water Rate', 'Oil Cum', 'Water Cum']  # misfit data blocks, in Misfit.blocks order
BLOCK_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
GROUPS = {'Provided simulation': COLORS['provided'], 'Acceptable match': '#9fc2ee', 'Selected case': COLORS['matched']}
METHODS = {'Sobol + L-BFGS-B': history_match, 'Differential Evolution': history_match_de,
           'ES-MDA': history_match_esmda}
METHOD_COLORS = {'Sobol + L-BFGS-B': COLORS['matched'], 'Differential Evolution': COLORS['alt'],
                 'ES-MDA': '#c2397a'}
SEEDS = [SEED, 1, 2]
N_CONVERGENCE_POINTS = 300  # log-spaced rows per run in the 'Convergence' sheet (the full trace is ~15,000 rows)

README_CASES = [
    ('Sheet', 'Content'),
    ('Case 1 … Case 10', 'One delivered case per sheet: 41 history quarters (1998-01-01 → 2008-01-01) then 80 forecast '
                         'quarters (2008-04-01 → 2028-01-01). Case 1 is the best history match.'),
    ('All Cases', 'The 10 case sheets stacked. Use this in Power BI with a slicer on Case.'),
    ('Parameters', 'One row per case: misfit, the 4 uncertain parameters and the 2028 cumulatives.'),
    ('', ''),
    ('Column', 'Meaning'),
    ('… - History', 'Observed field data (the same for every case). Filled on history rows only.'),
    ('… - Forecast', "This case's forecast. Also filled on 2008-01-01 (= last observed value) so the history and "
                     'forecast lines join when both columns are put on one line chart.'),
    ('… - Matched Model', "This case's history-matched proxy curve, 1998–2008: how well the case reproduces the history."),
    ('Rate', 'Average daily rate over the quarter ENDING at Date (STB/d for oil and water, MSCF/d for gas).'),
    ('Cum', 'Cumulative production at Date (STB for oil and water, MSCF for gas).'),
    ('Misfit', 'History-match misfit: mean squared mismatch in units of the assumed measurement error. Lower is better; '
               '1 = one measurement error away from the data on average.'),
    ('Period', '"History" or "Forecast", for filtering. Do not sum values across rows: each row is one date.'),
]

README_HM = [
    ('Sheet', 'Content'),
    ('Curves', '1998–2008 quarterly curves, one row per date and series. Source = "Observed field" (the target), '
               '"Provided simulation" (the 100 organiser cases) or "Selected case" (the matched model of the 10 '
               'delivered cases). Put Series on the legend/details of a line chart and colour by Source.'),
    ('Parameter Sets', 'Every scored parameter set, one row each. Group = "Provided simulation" (100 organiser cases), '
                       '"Acceptable match" (found by the search, misfit within 10 % of the best, not selected) or '
                       '"Selected case" (the 10 delivered cases). Groups do not overlap.'),
    ('', ''),
    ('Column', 'Meaning'),
    ('Misfit', 'Mean of the four block misfits below. Lower is better; 1 = one measurement error away from the data '
               'on average. Provided simulations are scored on their simulated curves, the others on the proxy.'),
    ('Misfit - <block>', 'Mean squared mismatch of one data block (oil rate, water rate, oil cum, water cum) in units of '
                         'its measurement error (rates 3 % of the peak observed rate, cums 2 % of the final observed '
                         'value). Divide by 4 to stack them into the total misfit. Gas is left out: it adds nothing '
                         'beyond oil.'),
    ('Rate / Cum', 'As in case_history_forecast.xlsx: rate = average over the quarter ENDING at Date.'),
]

README_OPT = [
    ('Sheet', 'Content'),
    ('Run Summary', 'One row per optimizer run (3 methods × 3 seeds): best misfit, cost, best parameters, 2028 P50.'),
    ('Convergence', 'Best misfit found so far vs number of proxy evaluations, per run (log-spaced sample of the full '
                    'trace). Plot with a log x-axis.'),
    ('Selected Cases', 'The 10 cases each run selects (best match first, no near-copies), with misfit and 2028 cums.'),
    ('Provided Cases', 'Misfit of the 100 organiser simulations, the baseline every method should beat.'),
    ('', ''),
    ('Note', 'All methods share the same misfit, parameter box and selection rule. "Run Time [s]" depends on the '
             'machine; "Proxy Evaluations" is the fair cost measure.'),
]


class CountedMisfit:
    """Wraps a Misfit and records every parameter set the method asks the proxy about, in order."""

    def __init__(self, misfit):
        self.misfit, self.f = misfit, []

    def __call__(self, raw):
        f = self.misfit(raw)
        self.f.append(f)
        return f

    def residuals(self, raw):  # ES-MDA asks for the residual curves instead of the misfit
        r = self.misfit.residuals(raw)
        self.f.append(np.mean(r ** 2, axis=1))
        return r

    def trace(self):
        return np.concatenate(self.f)


def case_sheet(k, case, field, fc):
    """One delivered case: history rows then forecast rows, one column per phase × quantity × source."""
    dates, n_h, n_f = field['dates'], len(field['dates']), len(FC_DATES)
    blank_h, blank_f = np.full(n_h, np.nan), np.full(n_f, np.nan)
    df = pd.DataFrame({'Case': f'Case {k + 1}', 'Date': dates.append(FC_DATES),
                       'Period': ['History'] * n_h + ['Forecast'] * n_f})
    for kind, label, obs, model in [('rate', 'Rate', field['obs_rate'], fc['rate_hist']),
                                    ('cum', 'Cum', field['obs_cum'], fc['cum_hist'])]:
        for v, (phase, unit) in enumerate(zip(PHASES, UNITS)):
            u = f'{unit}/d' if kind == 'rate' else unit
            fc_hist = blank_h.copy()
            fc_hist[-1] = obs[-1, v]  # join point: forecast starts from the last observed value (anchored forecast)
            df[f'{phase} {label} - History [{u}]'] = np.concatenate([obs[:, v], blank_f])
            df[f'{phase} {label} - Forecast [{u}]'] = np.concatenate([fc_hist, fc[kind][k, :, v]])
            df[f'{phase} {label} - Matched Model [{u}]'] = np.concatenate([model[k, :, v], blank_f])
    df['Misfit'] = case['misfit']
    for p in PARAM_COLS:
        df[p] = case[p]
    return df


def misfit_blocks(misfit, rate, cum):
    """(n, 4) misfit of each data block; their mean is the misfit."""
    curves = {'rate': rate, 'cum': cum}
    return np.column_stack([np.mean(((curves[kind][:, :, v] - d) / s) ** 2, axis=1) for v, kind, d, s in misfit.blocks])


def history_match_sheets(field, params, misfit, proxy, selected, accepted, fc):
    """'Curves' (observed, provided and matched 1998–2008 curves) and 'Parameter Sets' (every scored set)."""
    dates = field['dates']

    def curves(source, series, rate, cum, fit=np.nan):
        df = pd.DataFrame({'Date': dates, 'Source': source, 'Series': series, 'Misfit': fit})
        for kind, label, arr in [('rate', 'Rate', rate), ('cum', 'Cum', cum)]:
            for v, (phase, unit) in enumerate(zip(PHASES, UNITS)):
                df[f'{phase} {label} [{unit}/d]' if kind == 'rate' else f'{phase} {label} [{unit}]'] = arr[:, v]
        return df

    provided_blocks = misfit_blocks(misfit, field['rate'], field['cum'])
    curve_frames = [curves('Observed field', 'Observed field', field['obs_rate'], field['obs_cum'])]
    curve_frames += [curves('Provided simulation', f'Simulation {c}', field['rate'][c - 1], field['cum'][c - 1],
                            provided_blocks[c - 1].mean()) for c in params.index]
    curve_frames += [curves('Selected case', f'Case {k + 1}', fc['rate_hist'][k], fc['cum_hist'][k], row['misfit'])
                     for k, (_, row) in enumerate(selected.iterrows())]

    def sets(group, names, fits, blocks, raw):
        df = pd.DataFrame({'Group': group, 'Name': names, 'Misfit': fits})
        for b, name in enumerate(BLOCKS):
            df[f'Misfit - {name}'] = blocks[:, b]
        return pd.concat([df, pd.DataFrame(raw, columns=PARAM_COLS)], axis=1)

    is_selected = accepted[PARAM_COLS].apply(tuple, axis=1).isin(selected[PARAM_COLS].apply(tuple, axis=1))
    others = accepted[~is_selected].reset_index(drop=True)
    param_sets = pd.concat([
        sets('Provided simulation', [f'Simulation {c}' for c in params.index], provided_blocks.mean(1),
             provided_blocks, params[PARAM_COLS].to_numpy()),
        sets('Acceptable match', [f'Match {i + 1}' for i in range(len(others))], others['misfit'],
             misfit_blocks(misfit, *proxy.predict(others[PARAM_COLS].to_numpy())), others[PARAM_COLS].to_numpy()),
        sets('Selected case', [f'Case {k + 1}' for k in range(len(selected))], selected['misfit'].to_numpy(),
             misfit_blocks(misfit, fc['rate_hist'], fc['cum_hist']), selected[PARAM_COLS].to_numpy()),
    ], ignore_index=True)
    return pd.concat(curve_frames, ignore_index=True), param_sets


def write_sheets(path, sheets):
    """Write {name: DataFrame} with readable column widths, a frozen header and dates without times."""
    with pd.ExcelWriter(path, engine='openpyxl', datetime_format='yyyy-mm-dd') as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = 'A2'
            for j, col in enumerate(df.columns, 1):
                width = max(len(str(col)), *(len(str(x)) for x in df[col].head(50))) if len(df) else len(str(col))
                ws.column_dimensions[get_column_letter(j)].width = min(max(width + 2, 10), 70)


def run_optimizers(misfit, box, proxy, field, corr, provided):
    summary, convergence, selected_rows = [], [], []
    for method, search in METHODS.items():
        for seed in SEEDS:
            counted = CountedMisfit(misfit)
            t0 = time.perf_counter()
            sel, acc = search(counted, box, seed=seed)
            seconds = time.perf_counter() - t0
            f = counted.trace()
            best = np.minimum.accumulate(f)
            cum = forecast(proxy, sel[PARAM_COLS].to_numpy(), field['dates'], field['obs_rate'], field['obs_cum'],
                           corr)['cum'][:, -1]
            summary.append({'Method': method, 'Seed': seed, 'Best Misfit': best[-1],
                            'Best Provided Case Misfit': provided.min(),
                            'Improvement vs Best Provided Case [%]': 100 * (1 - best[-1] / provided.min()),
                            'Proxy Evaluations': len(f), 'Run Time [s]': round(seconds, 2),
                            'Acceptable Matches': len(acc),
                            **{f'Best {p}': sel[p].iloc[0] for p in PARAM_COLS},
                            **{f'{ph} Cum 2028 P50 [{u}]': np.median(cum[:, v])
                               for v, (ph, u) in enumerate(zip(PHASES, UNITS))}})
            idx = np.unique(np.geomspace(1, len(f), N_CONVERGENCE_POINTS).round().astype(int)) - 1
            convergence.append(pd.DataFrame({'Method': method, 'Seed': seed, 'Run': f'{method} (seed {seed})',
                                             'Proxy Evaluations': idx + 1, 'Best Misfit So Far': best[idx]}))
            for rank, (i, row) in enumerate(sel.iterrows(), 1):
                selected_rows.append({'Method': method, 'Seed': seed, 'Rank': rank, 'Misfit': row['misfit'],
                                      'Acceptance Limit': best[-1] * (1 + ACCEPT_TOL),
                                      **{p: row[p] for p in PARAM_COLS},
                                      **{f'{ph} Cum 2028 [{u}]': cum[rank - 1, v]
                                         for v, (ph, u) in enumerate(zip(PHASES, UNITS))}})
            print(f'  {method:24s} seed {seed:2d}: best misfit {best[-1]:.4f}, {len(f):6,} proxy evaluations, '
                  f'{seconds:5.1f}s')
    return pd.DataFrame(summary), pd.concat(convergence, ignore_index=True), pd.DataFrame(selected_rows)


def plot_case(df, fname):
    """Preview of one case page: rates (top) and cumulatives (bottom) for oil, gas, water."""
    case = df.iloc[0]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), sharex=True)
    for row, (label, scale, unit_fmt) in enumerate([('Rate', 1e3, 'thousand {}/d'), ('Cum', 1e6, 'MM{}')]):
        for col, (phase, unit) in enumerate(zip(PHASES, UNITS)):
            ax, u = axes[row, col], f'{unit}/d' if label == 'Rate' else unit
            ax.plot(df['Date'], df[f'{phase} {label} - History [{u}]'] / scale, color=COLORS['actual'], lw=2,
                    label='Observed history')
            ax.plot(df['Date'], df[f'{phase} {label} - Matched Model [{u}]'] / scale, color=COLORS['matched'],
                    lw=1.5, ls='--', label='Matched model (history)')
            ax.plot(df['Date'], df[f'{phase} {label} - Forecast [{u}]'] / scale, color=COLORS['matched'], lw=2,
                    label='Forecast')
            ax.axvline(FC_DATES[0] - pd.DateOffset(months=3), color=COLORS['muted'], lw=1, ls=':')
            ax.set_title(f'{phase} {"rate" if label == "Rate" else "cumulative"}')
            ax.set_ylabel(unit_fmt.format(unit))
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            if label == 'Cum':
                last = df[f'{phase} Cum - Forecast [{u}]'].iloc[-1] / scale
                ax.annotate(f'{last:.1f}', (df['Date'].iloc[-1], last), xytext=(4, 0), textcoords='offset points',
                            va='center', fontsize=8.5, color='#52514e')
    axes[0, 0].legend(loc='upper right')
    params = ' · '.join(f'{p} {case[p]:.3g}' for p in PARAM_COLS)
    fig.suptitle(f'{case["Case"]}: history + forecast to 2028\nmisfit {case["Misfit"]:.3f} · {params}',
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_hm_curves(curves, fname):
    """Preview of the history-match page: provided simulations vs selected cases vs the observed field."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), sharex=True)
    styles = {'Provided simulation': dict(color=COLORS['provided'], lw=0.7, alpha=0.7, zorder=1),
              'Selected case': dict(color=COLORS['matched'], lw=1.6, zorder=2)}
    for row, (label, scale, unit_fmt) in enumerate([('Rate', 1e3, 'thousand {}/d'), ('Cum', 1e6, 'MM{}')]):
        for col, (phase, unit) in enumerate(zip(PHASES, UNITS)):
            ax = axes[row, col]
            column = f'{phase} {label} [{unit}/d]' if label == 'Rate' else f'{phase} {label} [{unit}]'
            for source, style in styles.items():
                for j, (_, g) in enumerate(curves[curves['Source'] == source].groupby('Series', sort=False)):
                    n = curves.loc[curves['Source'] == source, 'Series'].nunique()
                    ax.plot(g['Date'], g[column] / scale, **style, label=f'{source} ({n})' if j == 0 else None)
            obs = curves[curves['Source'] == 'Observed field']
            ax.scatter(obs['Date'], obs[column] / scale, s=12, color=COLORS['actual'], zorder=3,
                       label='Observed field')
            ax.set_title(f'{phase} {"rate" if label == "Rate" else "cumulative"}')
            ax.set_ylabel(unit_fmt.format(unit))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[0, 0].legend(loc='lower center')
    fig.suptitle('History match, 1998–2008: provided simulations vs selected cases vs observed field',
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_hm_parameters(param_sets, fname):
    """Preview of the parameter page: which parameter values the history allows."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    rng = np.random.default_rng(0)
    for ax, p in zip(axes, PARAM_COLS):
        for y, (group, color) in enumerate(GROUPS.items()):
            df = param_sets[param_sets['Group'] == group]
            ax.scatter(df[p], y + rng.uniform(-0.15, 0.15, len(df)), s=40 if group == 'Selected case' else 14,
                       color=color, edgecolor='#fcfcfb', linewidth=0.8, zorder=3)
        labels = [f'{g} ({(param_sets["Group"] == g).sum()})' for g in GROUPS]
        ax.set_yticks(range(len(GROUPS)), labels if p == PARAM_COLS[0] else [])
        ax.set_ylim(-0.5, len(GROUPS) - 0.5)
        ax.set_title(p)
        ax.grid(axis='y', visible=False)
        if p == 'Permeability Multiplier':
            ax.set_xscale('log')
            ax.set_xticks([0.5, 1, 2, 5, 10], ['0.5', '1', '2', '5', '10'])
    fig.suptitle('What the production history constrains: porosity is pinned down, fault transmissibility is not',
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_misfit_breakdown(param_sets, fname, n_provided=10):
    """Preview of the misfit page: which data block each case misses, selected cases vs best provided ones."""
    best_provided = param_sets[param_sets['Group'] == 'Provided simulation'].nsmallest(n_provided, 'Misfit')
    df = pd.concat([param_sets[param_sets['Group'] == 'Selected case'], best_provided]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 6.4))
    y, left = np.arange(len(df)), np.zeros(len(df))
    for name, color in zip(BLOCKS, BLOCK_COLORS):
        width = df[f'Misfit - {name}'].to_numpy() / len(BLOCKS)  # stacked quarters sum to the misfit
        ax.barh(y, width, left=left, height=0.7, color=color, edgecolor='#fcfcfb', linewidth=2, label=name)
        left += width
    for yi, total in zip(y, df['Misfit']):
        ax.annotate(f'{total:.3f}', (total, yi), xytext=(4, 0), textcoords='offset points', va='center',
                    fontsize=8.5, color='#52514e')
    ax.set_yticks(y, df['Name'])
    xmax = df['Misfit'].max() * 1.12
    ax.axhline(n_provided - 0.5, color=COLORS['muted'], lw=1, ls=':')
    ax.annotate(f'↓ Best {n_provided} provided simulations', (xmax, n_provided - 0.5), xytext=(0, -4),
                textcoords='offset points', ha='right', va='top', fontsize=8.5, color='#52514e')
    ax.annotate('↑ Selected cases', (xmax, n_provided - 0.5), xytext=(0, 4), textcoords='offset points', ha='right',
                fontsize=8.5, color='#52514e')
    ax.grid(axis='y', visible=False)
    ax.set_xlim(0, xmax)
    ax.set_xlabel('Misfit (each block contributes a quarter)')
    ax.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, 1.0))
    ax.set_title('Where the misfit comes from: the water-rate block dominates every case', pad=28)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_optimizers(summary, convergence, selected, provided, fname):
    """Preview of the optimizer comparison page: convergence, best misfit, cost, selected-case misfits."""
    methods = list(METHODS)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.6))

    ax = axes[0, 0]
    for (method, seed), g in convergence.groupby(['Method', 'Seed'], sort=False):
        first = seed == SEEDS[0]
        ax.plot(g['Proxy Evaluations'], g['Best Misfit So Far'], color=METHOD_COLORS[method], lw=2 if first else 1,
                alpha=1 if first else 0.45, label=f'{method} (3 seeds)' if first else None)
    ax.axhline(provided.min(), color=COLORS['muted'], lw=1, ls='--')
    ax.annotate(f'Best provided simulation, case {provided.idxmin()} ({provided.min():.3f})',
                (convergence['Proxy Evaluations'].max(), provided.min()), xytext=(0, 4), textcoords='offset points',
                ha='right', fontsize=8.5, color='#52514e')
    ax.set_xscale('log')
    ax.set_ylim(0.435, 0.6)
    ax.set_xlabel('Proxy evaluations (log scale)')
    ax.set_ylabel('Best misfit found so far')
    ax.set_title('Convergence: best misfit vs proxy evaluations')
    ax.legend(loc='upper right')

    ax = axes[0, 1]
    for y, method in enumerate(methods):
        s = summary[summary['Method'] == method]
        ax.scatter(s['Best Misfit'], np.full(len(s), y), s=70, color=METHOD_COLORS[method], edgecolor='#fcfcfb',
                   linewidth=2, zorder=3)
        ax.annotate(f'{s["Best Misfit"].min():.4f} – {s["Best Misfit"].max():.4f}', (s['Best Misfit'].max(), y),
                    xytext=(10, 0), textcoords='offset points', va='center', fontsize=8.5, color='#52514e')
    ax.axvline(provided.min(), color=COLORS['muted'], lw=1, ls='--')
    ax.annotate(f'Best provided case\n{provided.min():.3f}', (provided.min(), len(methods) - 0.6), xytext=(-4, 0),
                textcoords='offset points', ha='right', fontsize=8.5, color='#52514e')
    ax.set_yticks(range(len(methods)), methods)
    ax.set_ylim(-0.6, len(methods) - 0.3)
    ax.set_xlim(0.440, 0.455)
    ax.invert_yaxis()
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Best misfit (lower is better; axis zoomed)')
    ax.set_title('Best misfit per run (3 seeds each)')

    ax = axes[1, 0]
    cost = summary.groupby('Method', sort=False)['Proxy Evaluations'].mean().reindex(methods)
    ax.barh(range(len(methods)), cost, height=0.55, color=[METHOD_COLORS[m] for m in methods])
    for y, c in enumerate(cost):
        ax.annotate(f'{c:,.0f}', (c, y), xytext=(4, 0), textcoords='offset points', va='center', fontsize=8.5,
                    color='#52514e')
    ax.set_yticks(range(len(methods)), methods)
    ax.invert_yaxis()
    ax.grid(axis='y', visible=False)
    ax.set_xlim(0, cost.max() * 1.15)
    ax.set_xlabel('Proxy evaluations (mean of 3 seeds)')
    ax.set_title('Cost: how many proxy runs each method needs')

    ax = axes[1, 1]
    rng = np.random.default_rng(0)
    main = selected[selected['Seed'] == SEEDS[0]]
    for y, method in enumerate(methods):
        s = main[main['Method'] == method]
        ax.scatter(s['Misfit'], y + rng.uniform(-0.12, 0.12, len(s)), s=40, color=METHOD_COLORS[method],
                   edgecolor='#fcfcfb', linewidth=1.2, zorder=3)
        ax.plot([s['Acceptance Limit'].iloc[0]] * 2, [y - 0.3, y + 0.3], color=COLORS['muted'], lw=1, ls=':')
    ax.annotate(f'Acceptance limit ({1 + ACCEPT_TOL:.2f} × best)', (main['Acceptance Limit'].iloc[0], -0.3),
                xytext=(-4, -2), textcoords='offset points', ha='right', va='top', fontsize=8.5, color='#52514e')
    ax.set_yticks(range(len(methods)), methods)
    ax.set_ylim(-0.6, len(methods) - 0.4)
    ax.invert_yaxis()
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Misfit')
    ax.set_title(f'Misfit of the {N_DELIVER} selected cases (seed {SEEDS[0]})')

    fig.suptitle('History-matching optimizers compared', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=130, bbox_inches='tight')
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
    for d in (RESULT_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    field, params = load_field(), load_params()
    dates = field['dates']
    corr = cum_correction(field['rate'][np.asarray(TRAIN_CASES) - 1], field['cum'][np.asarray(TRAIN_CASES) - 1],
                          dates)
    proxy = CurveProxy().fit(params, field['rate'], field['cum'])
    misfit, box = Misfit(proxy, field['obs_rate'], field['obs_cum']), ParamBox(params)
    provided = pd.Series(misfit.from_curves(field['rate'], field['cum']), index=params.index)

    print('[1/3] Case history + forecast workbook')
    selected, accepted = history_match(misfit, box, n_keep=N_DELIVER)  # same call as pipeline.main → same 10 cases
    fc = forecast(proxy, selected[PARAM_COLS].to_numpy(), dates, field['obs_rate'], field['obs_cum'], corr)
    cases = {f'Case {k + 1}': case_sheet(k, row, field, fc) for k, (_, row) in enumerate(selected.iterrows())}
    param_sheet = selected.rename(columns={'misfit': 'Misfit'})
    param_sheet.insert(0, 'Case', list(cases))
    for v, (phase, unit) in enumerate(zip(PHASES, UNITS)):
        param_sheet[f'{phase} Cum 2008 [{unit}]'] = field['obs_cum'][-1, v]
        param_sheet[f'{phase} Cum 2028 [{unit}]'] = fc['cum'][:, -1, v]
    write_sheets(CASES_PATH, {'Read Me': pd.DataFrame(README_CASES[1:], columns=README_CASES[0]), **cases,
                              'All Cases': pd.concat(cases.values(), ignore_index=True), 'Parameters': param_sheet})
    for name, df in cases.items():
        plot_case(df, f'case_{name.split()[-1]:0>2}_history_forecast.png')

    print('[2/3] History matching workbook')
    curves, param_sets = history_match_sheets(field, params, misfit, proxy, selected, accepted, fc)
    write_sheets(HM_PATH, {'Read Me': pd.DataFrame(README_HM[1:], columns=README_HM[0]), 'Curves': curves,
                           'Parameter Sets': param_sets})
    plot_hm_curves(curves, 'history_match_curves.png')
    plot_hm_parameters(param_sets, 'history_match_parameters.png')
    plot_misfit_breakdown(param_sets, 'history_match_misfit_breakdown.png')

    print('[3/3] Optimizer comparison workbook (3 methods × 3 seeds, ~2 minutes)')
    summary, convergence, sel_all = run_optimizers(misfit, box, proxy, field, corr, provided)
    provided_sheet = params.assign(Misfit=provided).reset_index().rename(columns={'case': 'Case'})
    provided_sheet = provided_sheet[['Case', 'Misfit', *PARAM_COLS]].sort_values('Misfit')
    write_sheets(OPT_PATH, {'Read Me': pd.DataFrame(README_OPT[1:], columns=README_OPT[0]), 'Run Summary': summary,
                            'Convergence': convergence, 'Selected Cases': sel_all, 'Provided Cases': provided_sheet})
    plot_optimizers(summary, convergence, sel_all, provided, 'optimizer_comparison.png')

    root = OUT_DIR.parent
    print(f'\nWrote {CASES_PATH.relative_to(root)}, {HM_PATH.relative_to(root)}, {OPT_PATH.relative_to(root)} and '
          f'{FIG_DIR.relative_to(root)}/ in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
