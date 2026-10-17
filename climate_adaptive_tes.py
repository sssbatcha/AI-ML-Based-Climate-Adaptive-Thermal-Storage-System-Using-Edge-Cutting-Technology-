#!/usr/bin/env python3
"""
AI/ML-Based Climate-Adaptive Thermal Energy Storage (TES) System
================================================================
End-to-end, runnable project (single file). Requires: numpy, pandas,
scikit-learn, scipy, matplotlib, joblib.

Pipeline
--------
1. Digital twin / data generator : 2 years of hourly weather, occupancy, tariff,
                                   and building cooling load (with climate warming
                                   trend + heatwaves => distribution drift).
2. ML forecaster                 : HistGradientBoosting day-ahead load forecast.
                                   * static model   (trained once on year 1)
                                   * adaptive model (weekly online retraining on a
                                     rolling window)  -> "climate-adaptive"
3. TES physical model            : chilled-water/ice tank with charge/discharge
                                   limits, efficiencies, standby loss and a
                                   temperature-dependent chiller COP.
4. Controllers                   : (a) No storage (baseline)
                                   (b) Rule-based schedule
                                   (c) ML-MPC  (forecast + receding-horizon LP)
                                   (d) Oracle MPC (perfect forecast, upper bound)
5. Edge deployment               : lightweight EdgeController (model + LP + safe
                                   fallback), model size and inference latency
                                   benchmark, JSON-style sensor payload demo.
6. Reporting                     : KPI table (cost, peak kW, energy, load shift)
                                   and a 6-panel result figure.

Usage
-----
    python climate_adaptive_tes.py                 # full run
    python climate_adaptive_tes.py --days 14 --out results
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# --------------------------------------------------------------------------- #
# 0. Configuration
# --------------------------------------------------------------------------- #


@dataclass
class TESConfig:
    # Tank (all thermal quantities in kWh_th / kW_th)
    capacity_kwh: float = 5000.0
    soc_min_frac: float = 0.05
    max_charge_kw: float = 600.0
    max_discharge_kw: float = 900.0
    eta_charge: float = 0.97
    eta_discharge: float = 0.97
    standby_loss_per_h: float = 0.001
    soc_init_frac: float = 0.30
    # Chiller
    chiller_cap_kw: float = 1400.0

    # Tariff (currency / kWh, currency / kW-month)
    price_offpeak: float = 0.07  # 22:00-06:00
    price_shoulder: float = 0.12
    price_peak: float = 0.28  # 17:00-22:00
    demand_charge: float = 12.0

    @property
    def soc_min(self) -> float:
        return self.soc_min_frac * self.capacity_kwh

    @property
    def soc_init(self) -> float:
        return self.soc_init_frac * self.capacity_kwh


HORIZON = 24  # MPC horizon (hours)


def chiller_cop(t_out):
    """Chiller COP falls as outdoor temperature rises (charging at night is cheaper)."""
    return np.clip(6.5 - 0.15 * (np.asarray(t_out, dtype=float) - 15.0), 2.5, 6.5)


def tariff(hours: np.ndarray, cfg: TESConfig) -> np.ndarray:
    p = np.full(hours.shape, cfg.price_shoulder, dtype=float)
    p[(hours >= 22) | (hours < 6)] = cfg.price_offpeak
    p[(hours >= 17) & (hours < 22)] = cfg.price_peak
    return p


# --------------------------------------------------------------------------- #
# 1. Digital twin: synthetic climate + load data
# --------------------------------------------------------------------------- #


def _ar1(n, phi, sigma, rng):
    x = np.zeros(n)
    e = rng.normal(0, sigma, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def generate_data(cfg: TESConfig, years: int = 2, seed: int = 42) -> pd.DataFrame:
    """Hourly dataset: weather (actual + forecast), occupancy, load, tariff, COP."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=24 * 365 * years, freq="h")
    n = len(idx)
    hour = idx.hour.values
    doy = idx.dayofyear.values
    t_years = np.arange(n) / (24 * 365)

    # --- Weather -------------------------------------------------------------
    seasonal = 4.0 * np.cos(2 * np.pi * (doy - 120) / 365)  # hottest ~ day 120
    diurnal = 5.0 * np.cos(2 * np.pi * (hour - 14) / 24)  # peak at 14:00
    warming = 2.0 * t_years  # climate trend (deg C / year)
    heatwave = np.zeros(n)
    for _ in range(3 * years):  # random 3-5 day heatwaves
        start = rng.integers(0, n - 24 * 6)
        heatwave[start : start + 24 * rng.integers(3, 6)] += rng.uniform(3, 5)
    heatwave = np.convolve(heatwave, np.ones(12) / 12, mode="same")  # smooth edges
    t_out = 28 + seasonal + diurnal + warming + heatwave + _ar1(n, 0.95, 0.35, rng)

    cloud = np.clip(0.25 + 0.2 * np.sin(2 * np.pi * (doy - 200) / 365) + _ar1(n, 0.97, 0.05, rng), 0, 0.85)
    solar = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None) * 900 * (1 - cloud)
    hum = np.clip(62 - 1.5 * (t_out - 28) + _ar1(n, 0.9, 1.5, rng), 30, 95)

    # Imperfect weather forecast (what the controller really sees)
    t_fc = t_out + _ar1(n, 0.9, 0.45, rng)
    solar_fc = np.clip(solar + rng.normal(0, 40, n), 0, None)

    # --- Occupancy & load -----------------------------------------------------
    weekday = idx.dayofweek.values < 5
    occ = np.where(weekday & (hour >= 8) & (hour < 19), 1.0, np.where(weekday & (hour >= 6) & (hour < 22), 0.35, 0.12))
    occ = np.clip(occ * (1 + rng.normal(0, 0.05, n)), 0.05, 1.2)

    # building ageing / growth: envelope leakage and internal gains rise over time (drift)
    envelope = 25.0 * (1 + 0.15 * t_years) * np.clip(t_out - 21, 0, None) * (0.4 + 0.6 * occ)
    load = (100 + envelope + 250 * (1 + 0.20 * t_years) * occ + 0.15 * solar + 2.0 * np.clip(hum - 60, 0, None)
            + _ar1(n, 0.7, 12, rng))
    load = np.clip(load, 40, 0.97 * (cfg.chiller_cap_kw + cfg.max_discharge_kw))

    df = pd.DataFrame(
        {
            "t_out": t_out, "solar": solar, "humidity": hum,
            "t_fc": t_fc, "solar_fc": solar_fc,
            "occupancy": occ, "load_kw": load,
            "price": tariff(hour, cfg), "cop": chiller_cop(t_out), "cop_fc": chiller_cop(t_fc),
        },
        index=idx,
    )
    return df


# --------------------------------------------------------------------------- #
# 2. ML forecaster (static vs. climate-adaptive)
# --------------------------------------------------------------------------- #

FEATURES = [
    "hour_sin", "hour_cos", "dow", "weekend", "t_fc", "solar_fc", "occupancy",
    "lag168", "lag_mean_24",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """All features are known >= 24 h ahead (weather forecast, schedule, lagged load),
    so one model serves every step of a 24 h receding horizon."""
    f = pd.DataFrame(index=df.index)
    h = df.index.hour.values
    f["hour_sin"] = np.sin(2 * np.pi * h / 24)
    f["hour_cos"] = np.cos(2 * np.pi * h / 24)
    f["dow"] = df.index.dayofweek.values
    f["weekend"] = (f["dow"] >= 5).astype(int)
    f["t_fc"] = df["t_fc"]
    f["solar_fc"] = df["solar_fc"]
    f["occupancy"] = df["occupancy"]
    f["lag168"] = df["load_kw"].shift(168)  # same hour last week
    f["lag_mean_24"] = df["load_kw"].shift(24).rolling(24).mean()  # yesterday's mean load
    return f[FEATURES]


def make_model(seed=0) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=250, learning_rate=0.08, max_depth=6, l2_regularization=1.0, random_state=seed
    )


def fit_model(X: pd.DataFrame, y: pd.Series) -> HistGradientBoostingRegressor:
    ok = X.notna().all(axis=1)
    return make_model().fit(X[ok], y[ok])


def adaptive_forecast(df, X, test_start, retrain_every_h=24 * 7, window_h=24 * 365):
    """Walk-forward forecasting. Adaptive model is retrained weekly on a rolling window
    using only data that has already been observed; static model is frozen at test_start."""
    n = len(df)
    y = df["load_kw"]
    static_model = fit_model(X.iloc[test_start - window_h : test_start], y.iloc[test_start - window_h : test_start])
    pred_static = np.full(n, np.nan)
    pred_adapt = np.full(n, np.nan)
    last_model = static_model
    retrains = 0
    for s in range(test_start, n, retrain_every_h):
        e = min(s + retrain_every_h, n)
        lo = max(0, s - window_h)
        last_model = fit_model(X.iloc[lo:s], y.iloc[lo:s])  # data strictly before s
        retrains += 1
        Xb = X.iloc[s:e].fillna(0)  # keep DataFrame (named columns)
        pred_adapt[s:e] = last_model.predict(Xb)
        pred_static[s:e] = static_model.predict(Xb)
    return static_model, last_model, pred_static, pred_adapt, retrains


# --------------------------------------------------------------------------- #
# 3. TES physical model
# --------------------------------------------------------------------------- #


def actuate(cfg: TESConfig, soc: float, load: float, g_plan: float, d_plan: float):
    """Apply a planned (charge, discharge) to the real tank/chiller with all limits.
    Returns (q_chiller, g, d, soc_next, unmet)."""
    avail_d = max(0.0, (soc - cfg.soc_min) * cfg.eta_discharge)
    room_g = max(0.0, (cfg.capacity_kwh - soc * (1 - cfg.standby_loss_per_h)) / cfg.eta_charge)
    d = min(d_plan, load, cfg.max_discharge_kw, avail_d)
    g = min(g_plan, cfg.max_charge_kw, room_g)
    q = load - d + g
    if q > cfg.chiller_cap_kw:  # chiller saturated: drop charging first, then lean on tank
        g = max(0.0, cfg.chiller_cap_kw - (load - d))
        q = load - d + g
        if q > cfg.chiller_cap_kw:
            extra = min(q - cfg.chiller_cap_kw, cfg.max_discharge_kw - d, avail_d - d)
            d += max(0.0, extra)
            q = load - d + g
    unmet = max(0.0, q - cfg.chiller_cap_kw)
    q = min(q, cfg.chiller_cap_kw)
    soc_next = soc * (1 - cfg.standby_loss_per_h) + cfg.eta_charge * g - d / cfg.eta_discharge
    return q, g, d, float(np.clip(soc_next, 0, cfg.capacity_kwh)), unmet


# --------------------------------------------------------------------------- #
# 4. Controllers
# --------------------------------------------------------------------------- #


def ctrl_baseline(ctx):
    return 0.0, 0.0


def ctrl_rule_based(ctx):
    """Charge during off-peak hours, discharge during the evening peak."""
    cfg, h, soc, load = ctx["cfg"], ctx["hour"], ctx["soc"], ctx["load"]
    if h >= 22 or h < 6:
        return cfg.max_charge_kw, 0.0
    if 17 <= h < 22:
        return 0.0, cfg.max_discharge_kw
    return 0.0, 0.0


def solve_mpc(cfg: TESConfig, soc0, load_fc, price, cop_fc):
    """Receding-horizon LP.
    Vars per hour t: q_t (chiller out), g_t (charge), d_t (discharge), s_t (SOC end of hour);
    plus P = peak chiller electric power over the horizon.
    min  sum price_t*q_t/cop_t + (demand_charge/30)*P - v*s_H      (v: terminal value of stored cold)
    s.t. q_t + d_t - g_t = L_t ;  s_t = (1-l)s_{t-1} + eta_c g_t - d_t/eta_d ;  q_t/cop_t <= P
    """
    H = len(load_fc)
    nq, ng, nd, ns = 0, H, 2 * H, 3 * H
    nP = 4 * H
    nv = 4 * H + 1
    cost = np.zeros(nv)
    cost[:H] = price / cop_fc
    cost[nP] = cfg.demand_charge / 30.0
    v = np.median(price) / np.median(cop_fc) * 0.9
    cost[ns + H - 1] = -v

    A_eq = np.zeros((2 * H, nv))
    b_eq = np.zeros(2 * H)
    for t in range(H):
        A_eq[t, nq + t], A_eq[t, nd + t], A_eq[t, ng + t] = 1, 1, -1
        b_eq[t] = load_fc[t]
        r = H + t
        A_eq[r, ns + t] = 1
        A_eq[r, ng + t] = -cfg.eta_charge
        A_eq[r, nd + t] = 1 / cfg.eta_discharge
        if t == 0:
            b_eq[r] = (1 - cfg.standby_loss_per_h) * soc0
        else:
            A_eq[r, ns + t - 1] = -(1 - cfg.standby_loss_per_h)
    A_ub = np.zeros((H, nv))
    for t in range(H):
        A_ub[t, nq + t] = 1 / cop_fc[t]
        A_ub[t, nP] = -1
    b_ub = np.zeros(H)

    bounds = ([(0, cfg.chiller_cap_kw)] * H + [(0, cfg.max_charge_kw)] * H +
              [(0, cfg.max_discharge_kw)] * H + [(cfg.soc_min, cfg.capacity_kwh)] * H + [(0, None)])
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return None
    x = res.x
    return x[ng], x[nd]  # first-hour charge, discharge


def make_mpc_controller(load_forecast: np.ndarray):
    def ctrl(ctx):
        t, cfg = ctx["t"], ctx["cfg"]
        df = ctx["df"]
        sl = slice(t, t + HORIZON)
        out = solve_mpc(cfg, ctx["soc"], load_forecast[sl], df["price"].values[sl], df["cop_fc"].values[sl])
        return out if out is not None else ctrl_rule_based(ctx)  # safe fallback

    return ctrl


# --------------------------------------------------------------------------- #
# Simulation + KPIs
# --------------------------------------------------------------------------- #


def simulate(df, cfg, controller, start, n_hours):
    soc = cfg.soc_init
    rec = []
    for t in range(start, start + n_hours):
        row = df.iloc[t]
        ctx = dict(cfg=cfg, df=df, t=t, hour=df.index[t].hour, soc=soc, load=row["load_kw"])
        g_plan, d_plan = controller(ctx)
        q, g, d, soc_next, unmet = actuate(cfg, soc, row["load_kw"], g_plan, d_plan)
        elec = q / row["cop"]
        rec.append((df.index[t], row["load_kw"], q, g, d, soc, elec, elec * row["price"], unmet, row["price"]))
        soc = soc_next
    return pd.DataFrame(rec, columns=["time", "load", "q", "charge", "discharge", "soc",
                                      "elec_kw", "energy_cost", "unmet", "price"]).set_index("time")


def kpis(sim: pd.DataFrame, cfg: TESConfig) -> dict:
    days = len(sim) / 24
    peak = sim["elec_kw"].max()
    demand = cfg.demand_charge * peak * days / 30
    energy = sim["energy_cost"].sum()
    h = sim.index.hour
    on_peak_kwh = sim.loc[(h >= 17) & (h < 22), "elec_kw"].sum()
    return {
        "energy_cost": energy, "demand_cost": demand, "total_cost": energy + demand,
        "peak_kw": peak, "elec_mwh": sim["elec_kw"].sum() / 1000,
        "on_peak_share_%": 100 * on_peak_kwh / sim["elec_kw"].sum(),
        "unmet_kwh_th": sim["unmet"].sum(),
    }


# --------------------------------------------------------------------------- #
# 5. Edge deployment
# --------------------------------------------------------------------------- #


class EdgeController:
    """Self-contained controller meant for a Raspberry Pi / Jetson / industrial gateway.
    Loads a serialized model, plans with the LP, and degrades gracefully (rule-based
    schedule) if the model or solver fails or if sensor data is invalid."""

    def __init__(self, model_path: str, cfg: TESConfig):
        self.cfg = cfg
        self.model = joblib.load(model_path)
        self.failures = 0

    def plan(self, payload: dict) -> dict:
        """payload: {soc_kwh, hour, features: 24xF list, price: [24], cop_fc: [24]}"""
        cfg = self.cfg
        try:
            soc = float(payload["soc_kwh"])
            if not (0 <= soc <= cfg.capacity_kwh * 1.02):
                raise ValueError("SOC out of range")
            X = pd.DataFrame(payload["features"], columns=FEATURES)
            load_fc = np.clip(self.model.predict(X), 0, None)
            out = solve_mpc(cfg, soc, load_fc, np.asarray(payload["price"]), np.asarray(payload["cop_fc"]))
            if out is None:
                raise RuntimeError("LP infeasible")
            g, d = out
            mode = "ml-mpc"
        except Exception:  # noqa: BLE001  (edge safety net)
            self.failures += 1
            g, d = ctrl_rule_based(dict(cfg=cfg, hour=int(payload.get("hour", 12)), soc=0, load=0))
            mode = "fallback-rule"
        return {"mode": mode, "charge_kw": round(float(g), 1), "discharge_kw": round(float(d), 1)}


def edge_benchmark(model, df, X, cfg, t0, out_dir, n_calls=30):
    path = os.path.join(out_dir, "tes_forecaster.joblib")
    joblib.dump(model, path, compress=3)
    size_kb = os.path.getsize(path) / 1024
    edge = EdgeController(path, cfg)
    lat = []
    for k in range(n_calls):
        t = t0 + k
        payload = {
            "soc_kwh": 2000.0, "hour": int(df.index[t].hour),
            "features": X.iloc[t : t + HORIZON].fillna(0).values.tolist(),
            "price": df["price"].values[t : t + HORIZON].tolist(),
            "cop_fc": df["cop_fc"].values[t : t + HORIZON].tolist(),
        }
        s = time.perf_counter()
        decision = edge.plan(payload)
        lat.append((time.perf_counter() - s) * 1000)
    # fault injection: corrupted sensor value must trigger the safe fallback
    bad = dict(payload, soc_kwh=-50.0)
    fallback = edge.plan(bad)
    return {
        "model_size_kb": round(size_kb, 1),
        "latency_ms_mean": round(float(np.mean(lat)), 1),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 1),
        "sample_decision": decision,
        "fault_injection_decision": fallback,
        "model_path": path,
    }


# --------------------------------------------------------------------------- #
# 6. Reporting
# --------------------------------------------------------------------------- #


def make_figure(df, test_start, ctrl_start, n_hours, pred_static, pred_adapt, sims, kp, path):
    fig, ax = plt.subplots(3, 2, figsize=(15, 13))

    # (a) forecast vs actual, one week
    w = slice(ctrl_start, ctrl_start + 24 * 7)
    ax[0, 0].plot(df.index[w], df["load_kw"].values[w], "k", lw=1.6, label="Actual load")
    ax[0, 0].plot(df.index[w], pred_adapt[w], "tab:green", lw=1.2, label="Adaptive ML")
    ax[0, 0].plot(df.index[w], pred_static[w], "tab:red", lw=1.0, ls="--", label="Static ML")
    ax[0, 0].set(title="(a) Day-ahead cooling load forecast", ylabel="kW_th")
    ax[0, 0].legend()

    # (b) rolling weekly MAE
    ts = pd.DataFrame({"y": df["load_kw"].values, "s": pred_static, "a": pred_adapt}, index=df.index).iloc[test_start:]
    mae_s = ts.groupby(pd.Grouper(freq="W")).apply(lambda g: mean_absolute_error(g.y, g.s))
    mae_a = ts.groupby(pd.Grouper(freq="W")).apply(lambda g: mean_absolute_error(g.y, g.a))
    ax[0, 1].plot(mae_s.index, mae_s.values, "tab:red", label="Static (frozen)")
    ax[0, 1].plot(mae_a.index, mae_a.values, "tab:green", label="Adaptive (weekly retrain)")
    ax[0, 1].set(title="(b) Forecast error under climate drift", ylabel="Weekly MAE (kW_th)")
    ax[0, 1].legend()

    # (c) SOC + price, 3 days, MPC
    s = sims["ML-MPC"].iloc[: 24 * 3]
    ax[1, 0].plot(s.index, s["soc"], "tab:blue", label="SOC (kWh_th)")
    ax[1, 0].set(title="(c) ML-MPC: tank state of charge", ylabel="kWh_th")
    a2 = ax[1, 0].twinx()
    a2.step(s.index, s["price"], "tab:orange", where="post", alpha=0.7)
    a2.set_ylabel("Tariff", color="tab:orange")
    ax[1, 0].legend(loc="upper left")

    # (d) electric demand profiles, 3 days
    for name, c in zip(["No storage", "Rule-based", "ML-MPC"], ["tab:gray", "tab:orange", "tab:green"]):
        ax[1, 1].plot(sims[name].index[:72], sims[name]["elec_kw"].values[:72], color=c, label=name)
    ax[1, 1].set(title="(d) Chiller electrical demand", ylabel="kW_el")
    ax[1, 1].legend()

    # (e) total cost
    names = list(kp.keys())
    cols = ["tab:gray", "tab:orange", "tab:green", "tab:blue"]
    ax[2, 0].bar(names, [kp[n]["total_cost"] for n in names], color=cols[: len(names)])
    ax[2, 0].set(title="(e) Total cost (energy + demand charge)", ylabel="Currency")
    ax[2, 0].tick_params(axis="x", rotation=15)

    # (f) peak demand
    ax[2, 1].bar(names, [kp[n]["peak_kw"] for n in names], color=cols[: len(names)])
    ax[2, 1].set(title="(f) Peak electrical demand", ylabel="kW_el")
    ax[2, 1].tick_params(axis="x", rotation=15)

    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.suptitle("Climate-Adaptive TES with ML Forecasting + MPC", fontsize=15, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Climate-adaptive ML thermal storage project")
    ap.add_argument("--days", type=int, default=30, help="control simulation length (days)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="tes_results", help="output directory")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = TESConfig()

    print("[1/6] Generating 2 years of hourly digital-twin data ...")
    df = generate_data(cfg, years=2, seed=args.seed)
    X = build_features(df)
    test_start = 24 * 365  # year 2 = deployment year (warmer than training year)

    print("[2/6] Training static + adaptive (walk-forward) forecasters ...")
    t0 = time.time()
    static_m, adapt_m, p_static, p_adapt, n_retrain = adaptive_forecast(df, X, test_start)
    y_te = df["load_kw"].values[test_start:]
    mae_s = mean_absolute_error(y_te, p_static[test_start:])
    mae_a = mean_absolute_error(y_te, p_adapt[test_start:])
    print(f"      Year-2 MAE  static: {mae_s:6.1f} kW | adaptive: {mae_a:6.1f} kW "
          f"({100 * (1 - mae_a / mae_s):.1f}% better, {n_retrain} retrains, {time.time() - t0:.0f}s)")

    # Control window: hottest period of the deployment year (centered on day ~120)
    ctrl_start = test_start + 24 * (120 - args.days // 2)
    n_hours = 24 * args.days
    print(f"[3/6] Simulating {args.days}-day control window starting {df.index[ctrl_start].date()} ...")

    controllers = {
        "No storage": ctrl_baseline,
        "Rule-based": ctrl_rule_based,
        "ML-MPC": make_mpc_controller(np.clip(p_adapt, 0, None)),
        "Oracle MPC": make_mpc_controller(df["load_kw"].values),
    }
    sims, kp = {}, {}
    for name, ctrl in controllers.items():
        s = time.time()
        sims[name] = simulate(df, cfg, ctrl, ctrl_start, n_hours)
        kp[name] = kpis(sims[name], cfg)
        print(f"      {name:<11} done ({time.time() - s:4.1f}s)")

    print("[4/6] Results")
    table = pd.DataFrame(kp).T
    base = table.loc["No storage", "total_cost"]
    table["saving_%"] = 100 * (1 - table["total_cost"] / base)
    base_pk = table.loc["No storage", "peak_kw"]
    table["peak_cut_%"] = 100 * (1 - table["peak_kw"] / base_pk)
    with pd.option_context("display.float_format", "{:,.1f}".format, "display.width", 200):
        print(table[["energy_cost", "demand_cost", "total_cost", "saving_%", "peak_kw", "peak_cut_%",
                     "elec_mwh", "on_peak_share_%", "unmet_kwh_th"]])
    table.to_csv(os.path.join(args.out, "kpi_summary.csv"))

    print("[5/6] Edge deployment benchmark ...")
    edge = edge_benchmark(adapt_m, df, X, cfg, ctrl_start, args.out)
    print(f"      model {edge['model_size_kb']} KB | latency mean {edge['latency_ms_mean']} ms, "
          f"p95 {edge['latency_ms_p95']} ms")
    print(f"      normal decision : {edge['sample_decision']}")
    print(f"      fault injection : {edge['fault_injection_decision']}")

    print("[6/6] Saving figure and report ...")
    make_figure(df, test_start, ctrl_start, n_hours, p_static, p_adapt, sims, kp,
                os.path.join(args.out, "tes_results.png"))
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump({"config": asdict(cfg), "forecast_mae": {"static": mae_s, "adaptive": mae_a},
                   "kpis": kp, "edge": {k: v for k, v in edge.items() if k != "model_path"}}, f, indent=2, default=float)
    print(f"Done. Outputs in ./{args.out}/")


if __name__ == "__main__":
    main()
