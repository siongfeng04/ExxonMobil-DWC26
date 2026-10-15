"""Optimisation (history matching): find parameter sets whose proxy history reproduces the observed field data.

Misfit = mean over four data blocks (oil rate, water rate, oil cum, water cum) of the mean squared residual in units
of an assumed measurement error σ. A value of 1 means the proxy is, on average, one σ away from the data.
Gas is left out: GOR is ~0.36 MSCF/STB in every case and quarter, so gas carries no information beyond oil.

Search: Sobol screening of the whole parameter box, then L-BFGS-B polishing from the best distinct starts.
history_match_de (differential evolution) and history_match_esmda (ensemble smoother) are the alternative searches,
compared in notebook/02_History_Matching_Forecast.
The parameter box is the range spanned by all 100 simulated cases (train, validation and test alike), so the
candidates come from the full parameter population and the proxy is never asked to extrapolate.

Selection of the deliverable cases (DWC26 FAQ 8–9: driven by history-match quality, uncertainty second): every point
within ACCEPT_TOL of the best misfit is an acceptable match. From those, the cases are taken best match first,
skipping any within MIN_SEPARATION of a case already picked. Without that skip the 10 best are near-copies of one
point (they differ only in fault transmissibility, which the history leaves unconstrained and which barely changes
the forecast), so the forecasts would show no parameter uncertainty at all.
"""
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm, qmc

from .config import PARAM_COLS, SEED

SIGMA_RATE = 0.03  # × max observed rate of that phase
SIGMA_CUM = 0.02   # × final observed cumulative of that phase
ACCEPT_TOL = 0.10  # acceptable match: misfit ≤ (1 + ACCEPT_TOL) × best misfit
MIN_SEPARATION = 0.2  # selected cases are at least this far apart in the unit parameter cube (no near-copies)
ESMDA_ALPHAS = (9.333, 7.0, 4.0, 2.0)  # inflation factors; sum of 1/α = 1 (Emerick & Reynolds, 2013)
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

    def residuals(self, raw):
        """(n, 4T) residuals in units of σ, blocks concatenated. All blocks have T dates, so the misfit is the
        mean of their squares."""
        rate, cum = self.proxy.predict(np.atleast_2d(raw))
        curves = {'rate': rate, 'cum': cum}
        return np.concatenate([(curves[kind][:, :, v] - d) / s for v, kind, d, s in self.blocks], axis=1)


def _distinct(U, order, k, min_dist):
    """Greedy pick of up to k rows of U (in `order`), each at least min_dist from those already picked."""
    picked = []
    for i in order:
        if all(np.linalg.norm(U[i] - U[j]) >= min_dist for j in picked):
            picked.append(i)
            if len(picked) == k:
                break
    return picked


def history_match(misfit, box, n_sobol_log2=13, n_starts=40, n_keep=10, accept_tol=ACCEPT_TOL, seed=SEED):
    """Returns (selected, accepted): DataFrames of [misfit, *PARAM_COLS].
    selected: the n_keep best acceptable matches that are not near-copies, sorted by misfit (row 0 = best match).
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


def history_match_esmda(misfit, box, n_ens=200, alphas=ESMDA_ALPHAS, n_keep=10, accept_tol=ACCEPT_TOL, seed=SEED):
    """Same contract as history_match, with the ensemble smoother with multiple data assimilation (ES-MDA) as the search.

    An ensemble of n_ens parameter sets is drawn from the prior and updated len(alphas) times with a Kalman-type step,
    z ← z + C_zr (C_rr + α I)⁻¹ (√α ε − r), where r are the σ-scaled residuals (so the data covariance is I) and the
    covariances come from the ensemble itself: no gradients and no search. Parameters live in Gaussian space z and
    map to the unit cube by u = Φ(z), so z ~ N(0, I) is the uniform prior on the box and updates never leave it.
    Needs misfit.residuals. Every ensemble (prior, intermediates, final) is scored and goes to the selection step."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_ens, 4))
    seen_U, seen_f = [], []
    for alpha in (*alphas, None):  # None: score the final ensemble, no update
        U = norm.cdf(Z)
        R = misfit.residuals(box.to_raw(U))
        seen_U.append(U)
        seen_f.append(np.mean(R ** 2, axis=1))
        if alpha is None:
            break
        dZ, dR = Z - Z.mean(0), R - R.mean(0)
        C_zr, C_rr = dZ.T @ dR / (n_ens - 1), dR.T @ dR / (n_ens - 1)
        noise = np.sqrt(alpha) * rng.standard_normal(R.shape)
        Z = Z + (noise - R) @ np.linalg.solve(C_rr + alpha * np.eye(R.shape[1]), C_zr.T)
    return _select(np.vstack(seen_U), np.concatenate(seen_f), box, n_keep, accept_tol)


def _select(U_all, f_all, box, n_keep, accept_tol):
    order = np.argsort(f_all, kind='stable')
    U_all, f_all = U_all[order], f_all[order]
    ok = f_all <= f_all[0] * (1 + accept_tol)

    def frame(idx):
        out = pd.DataFrame(box.to_raw(U_all[idx]), columns=PARAM_COLS)
        out.insert(0, 'misfit', f_all[idx])
        return out

    keep = _distinct(U_all[ok], range(ok.sum()), n_keep, MIN_SEPARATION)  # best first, skipping near-copies
    return frame(np.flatnonzero(ok)[keep]), frame(np.flatnonzero(ok))
