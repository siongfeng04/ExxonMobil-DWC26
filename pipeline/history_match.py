"""Optimisation (history matching): find parameter sets whose proxy history reproduces the observed field data.

Misfit = mean over four data blocks (oil rate, water rate, oil cum, water cum) of the mean squared residual in units
of an assumed measurement error σ. A value of 1 means the proxy is, on average, one σ away from the data.
Gas is left out: GOR is ~0.36 MSCF/STB in every case and quarter, so gas carries no information beyond oil.

Search: Sobol screening of the whole parameter box, then L-BFGS-B polishing from the best distinct starts.
history_match_de is the alternative search (differential evolution), compared in notebook/History_Matching_Forecast.
The parameter box is the range spanned by the 100 simulated cases, so the proxy is never asked to extrapolate.

Selection of the deliverable cases: the misfit has a floor (~0.44) that no parameter set gets below, because the
observed late-time water cut lies outside what the simulated ensemble can produce. Differences of a few percent
above that floor are not meaningful, so every point within ACCEPT_TOL of the best is an acceptable match. From those,
the cases are spread out with farthest-point sampling (seeded with the best match), so the forecasts express the
remaining parameter uncertainty rather than ten copies of one answer. Fault transmissibility, for example, is almost
unconstrained by the history.
"""
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc

from .config import PARAM_COLS, SEED

SIGMA_RATE = 0.03  # × max observed rate of that phase
SIGMA_CUM = 0.02   # × final observed cumulative of that phase
ACCEPT_TOL = 0.10  # acceptable match: misfit ≤ (1 + ACCEPT_TOL) × best misfit
LOG_SCALED = np.array([False, False, True, False])  # permeability multiplier is searched on a log scale


class ParamBox:
    """Maps the unit cube [0, 1]^4 to raw parameter values (log scale for permeability)."""

    def __init__(self, params):
        lo, hi = params[PARAM_COLS].min().to_numpy(float), params[PARAM_COLS].max().to_numpy(float)
        self.lo = np.where(LOG_SCALED, np.log(lo), lo)
        self.hi = np.where(LOG_SCALED, np.log(hi), hi)

    def to_raw(self, u):
        v = self.lo + np.atleast_2d(u) * (self.hi - self.lo)
        return np.where(LOG_SCALED, np.exp(v), v)

    def to_unit(self, raw):
        v = np.where(LOG_SCALED, np.log(np.atleast_2d(raw)), np.atleast_2d(raw))
        return (v - self.lo) / (self.hi - self.lo)


class Misfit:
    def __init__(self, proxy, obs_rate, obs_cum):
        """obs_rate/obs_cum: (T, 3) observations in simulator convention, same dates as the proxy."""
        self.proxy = proxy
        self.blocks = [(0, 'rate', obs_rate[:, 0], SIGMA_RATE * obs_rate[:, 0].max()),
                       (2, 'rate', obs_rate[:, 2], SIGMA_RATE * obs_rate[:, 2].max()),
                       (0, 'cum', obs_cum[:, 0], SIGMA_CUM * obs_cum[-1, 0]),
                       (2, 'cum', obs_cum[:, 2], SIGMA_CUM * obs_cum[-1, 2])]

    def from_curves(self, rate, cum):
        """rate/cum: (n, T, 3) → misfit (n,)"""
        curves = {'rate': rate, 'cum': cum}
        return np.mean([np.mean(((curves[kind][:, :, v] - d) / s) ** 2, axis=1)
                        for v, kind, d, s in self.blocks], axis=0)

    def __call__(self, raw):
        return self.from_curves(*self.proxy.predict(np.atleast_2d(raw)))


def _distinct(U, order, k, min_dist):
    """Greedy pick of up to k rows of U (in `order`), each at least min_dist from those already picked."""
    picked = []
    for i in order:
        if all(np.linalg.norm(U[i] - U[j]) >= min_dist for j in picked):
            picked.append(i)
            if len(picked) == k:
                break
    return picked


def _farthest_points(U, k):
    """Row 0 first, then repeatedly the row farthest from everything picked so far."""
    picked = [0]
    d = np.linalg.norm(U - U[0], axis=1)
    while len(picked) < min(k, len(U)):
        i = int(np.argmax(d))
        picked.append(i)
        d = np.minimum(d, np.linalg.norm(U - U[i], axis=1))
    return picked


def history_match(misfit, box, n_sobol_log2=13, n_starts=40, n_keep=10, accept_tol=ACCEPT_TOL, seed=SEED):
    """Returns (selected, accepted): DataFrames of [misfit, *PARAM_COLS].
    selected: n_keep spread-out acceptable matches, sorted by misfit (row 0 = best match).
    accepted: every evaluated point within accept_tol of the best misfit."""
    U = qmc.Sobol(4, scramble=True, seed=seed).random_base2(n_sobol_log2)
    f = np.concatenate([misfit(box.to_raw(chunk)) for chunk in np.array_split(U, max(1, len(U) // 1024))])

    starts = _distinct(U, np.argsort(f), n_starts, min_dist=0.05)
    U_opt, f_opt = [], []
    for i in starts:
        res = minimize(lambda u: float(misfit(box.to_raw(u))[0]), U[i], method='L-BFGS-B', bounds=[(0, 1)] * 4)
        U_opt.append(res.x)
        f_opt.append(res.fun)
    return _select(np.vstack([U_opt, U]), np.concatenate([f_opt, f]), box, n_keep, accept_tol)


def history_match_de(misfit, box, popsize=15, maxiter=300, n_keep=10, accept_tol=ACCEPT_TOL, seed=SEED):
    """Same contract as history_match, with scipy's differential evolution as the search.

    DE evolves a population (popsize × 4 = 60 members) by mixing the differences between members, keeping a trial
    only if it beats its parent, and finishes with an L-BFGS-B polish of the best member. The whole population is
    scored in one proxy call per generation (vectorized). Every point DE evaluates is a candidate for the
    acceptable set, so the selection step works exactly as in history_match."""
    seen_U, seen_f = [], []

    def objective(u):  # u is (4, S) for a generation, (4,) during the final polish
        u = np.atleast_2d(u.T)
        f = misfit(box.to_raw(u))
        seen_U.append(u)
        seen_f.append(f)
        return f

    differential_evolution(objective, [(0, 1)] * 4, popsize=popsize, maxiter=maxiter, tol=1e-8, seed=seed,
                           polish=True, vectorized=True, updating='deferred')
    return _select(np.vstack(seen_U), np.concatenate(seen_f), box, n_keep, accept_tol)


def _select(U_all, f_all, box, n_keep, accept_tol):
    order = np.argsort(f_all, kind='stable')
    U_all, f_all = U_all[order], f_all[order]
    ok = f_all <= f_all[0] * (1 + accept_tol)

    def frame(idx):
        out = pd.DataFrame(box.to_raw(U_all[idx]), columns=PARAM_COLS)
        out.insert(0, 'misfit', f_all[idx])
        return out

    keep = sorted(_farthest_points(U_all[ok], n_keep))  # indices into the misfit-sorted accepted set
    return frame(np.flatnonzero(ok)[keep]), frame(np.flatnonzero(ok))
