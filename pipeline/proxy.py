"""History proxy and decline-curve forecast, ported from notebook/ML_Proxy_Model.ipynb (sections 4-6).

The notebook selected the Gaussian Process proxy (validation cum NRMSE < 0.1 %) and Approach A (proxy + decline)
as the forecast method on the end-to-end backtest; only those two are kept here.
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from .config import PARAM_COLS, SEED


def make_features(p):
    X = p[PARAM_COLS].to_numpy(float).copy() if isinstance(p, pd.DataFrame) else np.array(p, float, ndmin=2)
    X[:, 2] = np.log(X[:, 2])  # permeability multiplier ~1–10× → log scale
    return X


def pack(rate, cum):
    n = len(rate)
    return np.concatenate([cum.reshape(n, -1), rate.reshape(n, -1)], axis=1)


def unpack(Y, T):
    n = len(Y)
    cum = Y[:, :T * 3].reshape(n, T, 3)
    rate = np.clip(Y[:, T * 3:].reshape(n, T, 3), 0, None)
    cum = np.maximum.accumulate(np.clip(cum, 0, None), axis=1)
    return rate, cum


def make_gp():
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=np.ones(4), length_scale_bounds=(1e-2, 1e3), nu=2.5)
              + WhiteKernel(1e-5, (1e-10, 1e-1)))
    return GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=4, random_state=SEED)


class CurveProxy:
    """Raw parameters → quarterly rate & cumulative curves via block-scaled PCA + one GP per PCA score."""

    def __init__(self, var_kept=0.99999):
        self.var_kept = var_kept

    def fit(self, params, rate, cum):
        self.x_scaler = StandardScaler().fit(make_features(params))
        X = self.x_scaler.transform(make_features(params))
        self.T = rate.shape[1]
        Y = pack(rate, cum)
        self.mu = Y.mean(axis=0)
        Yc = Y - self.mu
        block = np.concatenate([np.tile(np.arange(3), self.T), 3 + np.tile(np.arange(3), self.T)])
        self.scale = np.ones(Y.shape[1])
        for b in range(6):
            self.scale[block == b] = np.sqrt((Yc[:, block == b] ** 2).mean())
        self.pca = PCA(n_components=self.var_kept, svd_solver='full').fit(Yc / self.scale)
        S = self.pca.transform(Yc / self.scale)
        self.models = [make_gp().fit(X, S[:, k]) for k in range(S.shape[1])]
        return self

    def predict(self, params):
        """params: DataFrame with PARAM_COLS or (n, 4) array of raw values. Returns (rate, cum), each (n, T, 3)."""
        X = self.x_scaler.transform(make_features(params))
        S = np.column_stack([m.predict(X) for m in self.models])
        return unpack(self.pca.inverse_transform(S) * self.scale + self.mu, self.T)


def cum_correction(rate, cum, dates, late=12):
    """Median ratio Δcum / (rate · days) over the last `late` quarters (right-point rule calibration)."""
    days = np.diff(dates.values).astype('timedelta64[D]').astype(float)
    inc = np.diff(cum, axis=1)[:, -late:]
    right = (rate[:, 1:] * days[None, :, None])[:, -late:]
    return np.nanmedian((inc / right).reshape(-1, 3), axis=0)


def integrate_cum(rate_fc, cum_anchor, anchor_date, fc_dates, corr):
    """Right-point integration of forecast rates (rate at t = average over the preceding period)."""
    days = np.diff(pd.DatetimeIndex([anchor_date]).append(pd.DatetimeIndex(fc_dates)).values)
    days = days.astype('timedelta64[D]').astype(float)
    return cum_anchor[:, None, :] + np.cumsum(rate_fc * days[None, :, None] * corr, axis=1)


def _log_arps(t, log_qi, log_di, b):
    return log_qi - np.log1p(b * np.exp(log_di) * t) / b


def decline_forecast(rate_hist, cum_hist, hist_dates, fc_dates, window, corr):
    """Approach A: Arps on oil rate, exponential on liquid rate, water = liquid − oil, gas = last GOR × oil.
    rate_hist/cum_hist: (n, T, 3) ending at hist_dates[-1]. Returns (rate_fc, cum_fc, fit_params)."""
    anchor = hist_dates[-1]
    t_hist = np.asarray((hist_dates - anchor).days, float) / 365.25
    t_fc = np.asarray((pd.DatetimeIndex(fc_dates) - anchor).days, float) / 365.25
    w = slice(len(hist_dates) - window, None)
    tw = t_hist[w] - t_hist[w][0]
    rate_fc = np.zeros((len(rate_hist), len(fc_dates), 3))
    fits = []
    for i in range(len(rate_hist)):
        qo, qw = rate_hist[i, :, 0], rate_hist[i, :, 2]
        ql = qo + qw
        (lqi, ldi, b), _ = curve_fit(_log_arps, tw, np.log(np.clip(qo[w], 1.0, None)),
                                     p0=(np.log(qo[w][0]), np.log(0.1), 0.5),
                                     bounds=([-np.inf, np.log(1e-4), 1e-3], [np.inf, np.log(5.0), 1.0]), maxfev=20000)
        shift = -t_hist[w][0]  # anchor date expressed in window time
        qo_fc = qo[-1] * np.exp(_log_arps(t_fc + shift, lqi, ldi, b) - _log_arps(shift, lqi, ldi, b))
        dl = float(np.clip(-np.polyfit(tw, np.log(ql[w]), 1)[0], -0.02, 1.0))
        ql_fc = ql[-1] * np.exp(-dl * t_fc)
        qo_fc = np.minimum(qo_fc, ql_fc)
        last_gor = rate_hist[i, -1, 1] / max(qo[-1], 1e-9)
        rate_fc[i] = np.stack([qo_fc, last_gor * qo_fc, ql_fc - qo_fc], axis=-1)
        fits.append({'oil_Di_per_yr': np.exp(ldi), 'oil_b': b, 'liquid_D_per_yr': dl, 'GOR': last_gor})
    return rate_fc, integrate_cum(rate_fc, cum_hist[:, -1], anchor, fc_dates, corr), pd.DataFrame(fits)
