"""Earn-rate forecaster for gta6-trophy-rarity-forecast-01 (Python 3.13, stdlib only).

Two models are blended, because a rate is a ratio and the two sides fail
differently. The index model works on logit rates directly: logit(rate) is a
shared progress index minus a per-trophy difficulty, with a per-trophy slope
on the index shrunk toward its class mean; the index is extrapolated by an
ensemble of trend fits. The earner model works on the ratio's parts: players
tracked follows an ensemble of growth curves, and each trophy's earners
follow a saturating curve in log-time whose location and ceiling are fitted
per trophy but regularised by pseudo-observations at the target implied by a
prior read off the trophy's metadata (chapter number, counter size, tier).
The blend widens by how much the two medians disagree, so a world where one
model is structurally wrong costs spread, not the forecast. Structural caps
keep the platinum and the 100%-completion trophy at or below every trophy
they presuppose, and later story chapters at or below earlier ones. A case
whose page never changed in its history (a stale launch-night page) takes a
dilution branch instead: at launch the tracked base broadens far faster than
earners accumulate, so rates fall, and the branch prices that in.

Runs once per case: reads every *.in in $MAKEFUN_INPUT_DIR (else /input) and
writes <id>.out to $MAKEFUN_OUTPUT_DIR (else /output). Deterministic: fixed
iteration counts, no randomness, sorted iteration everywhere.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

IN_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_OF_LEVEL = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------- small maths

def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def phi(x):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def round4(x):
    v = math.floor(x * 10000.0 + 0.5) / 10000.0
    return int(v) if float(v).is_integer() else v


def parse_time(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def wls_line(xs, ys, ws):
    """Weighted least squares y ~ a + b x; returns (a, b, rmse)."""
    sw = sum(ws)
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx if sxx > 1e-12 else 0.0
    a = my - b * mx
    rmse = math.sqrt(sum(w * (y - a - b * x) ** 2 for x, y, w in zip(xs, ys, ws)) / sw)
    return a, b, rmse


# ------------------------------------------------------- trophy classification

CHAPTER_RE = re.compile(r"[Cc]hapter\s+(\d+)")
COUNT_RE = re.compile(r"(\d[\d,]*)")

# Prior for the earners curve E(u) = K * Phi((ln u - mu) / sigma): mu is the
# log-time by which half the eventual earners hold the trophy, share is K as
# a fraction of the players tracked at the target.
PRIORS = {
    "chapter":    (3.55, 0.60),
    "finale":     (4.75, 0.42),
    "mission":    (4.35, 0.38),
    "misc":       (4.50, 0.30),
    "grind":      (5.30, 0.20),
    "completion": (5.90, 0.055),
    "platinum":   (6.10, 0.030),
}


def classify(trophy):
    """Sort a trophy into a pace class from its own metadata."""
    desc = trophy.get("description") or ""
    tier = trophy.get("tier") or ""
    if tier == "platinum":
        return "platinum", None
    if "100% Completion" in desc:
        return "completion", None
    m = CHAPTER_RE.search(desc)
    if m:
        return "chapter", int(m.group(1))
    low = desc.lower()
    if "final mission" in low and tier == "gold":
        return "finale", None
    if desc.startswith("Complete "):
        return "mission", None
    # Counters and full-map sweeps grind slowly: a stated count of 8+, a price
    # tag of $10,000+, or an "every/all" sweep in the description.
    for raw in COUNT_RE.findall(desc):
        val = int(raw.replace(",", ""))
        if val >= 8 or (desc.find("$") >= 0 and val >= 10000):
            return "grind", None
    if " every " in low or low.startswith("earn a gold medal in all"):
        return "grind", None
    return "misc", None


def prior_for(cls, chapter_no):
    mu, share = PRIORS[cls]
    if cls == "chapter" and chapter_no:
        mu = mu + 0.28 * (chapter_no - 1)
    return mu, share


# ------------------------------------------------------------- page modelling

def compress_states(history):
    """Collapse runs of identical reads into page states.

    A state is one distinct page: (time of first read showing it, players
    tracked, rates). Exophase refreshes in steps, so the distinct states are
    the observations; the repeats between refreshes carry no new signal.
    """
    states = []
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if states and states[-1][0] == key:
            continue
        states.append((key, read))
    return [(parse_time(r["slot"]), r["players_tracked"], list(r["rates"])) for _, r in states]


# --- the index model ---------------------------------------------------------

def two_way(states, u_of, active):
    """Fit logit(rate_ij) = g_j - d_i over the distinct states.

    Alternating least squares with recency weights on the states (36 h
    half-life, floored), so the stale launch page counts for little and the
    recent page states set the index. Returns (g, weights, ys): one index
    value per state and the clamped logit rates per active trophy.
    """
    m = len(states)
    t_last = states[-1][0]
    ws = []
    for (t, _n, _r) in states:
        age_h = (t_last - t).total_seconds() / 3600.0
        ws.append(max(0.05, math.exp(-age_h / 36.0)))
    ys = {}
    for i in active:
        col = []
        for (_t, _n, rates) in states:
            r = rates[i]
            col.append(None if r is None else logit(r))
        ys[i] = col
    g = []
    for j in range(m):
        vals = [ys[i][j] for i in active if ys[i][j] is not None]
        g.append(sum(vals) / len(vals) if vals else 0.0)
    d = {i: 0.0 for i in active}
    for _ in range(60):
        for i in active:
            num = 0.0
            den = 0.0
            for j in range(m):
                if ys[i][j] is not None:
                    num += ws[j] * (g[j] - ys[i][j])
                    den += ws[j]
            if den > 0:
                d[i] = num / den
        centre = sum(d.values()) / len(d)
        for i in active:
            d[i] -= centre
        g = [gj + centre for gj in g]
        for j in range(m):
            vals = [ys[i][j] + d[i] for i in active if ys[i][j] is not None]
            if vals:
                g[j] = sum(vals) / len(vals)
    return g, ws, ys


def index_trend(states, u_of, g, ws, u_target):
    """Extrapolate the progress index to the target; returns (g_target, sigma_g).

    Three candidates -- a recency-weighted line in ln u over all states, a
    line over the last 48 clock hours of states, and no movement at all --
    combined by their median, with sigma from their spread and the trend
    fit's residual grown by the log-distance extrapolated.
    """
    xs = [math.log(u_of(t)) for (t, _n, _r) in states]
    xt = math.log(u_target)
    cands = []
    a, b, rmse = wls_line(xs, g, ws)
    cands.append(a + b * xt)
    recent = [(x, gj, w) for x, gj, w, (t, _n, _r) in zip(xs, g, ws, states)
              if (states[-1][0] - t).total_seconds() <= 48 * 3600]
    if len(recent) >= 3:
        a2, b2, _r2 = wls_line([r[0] for r in recent], [r[1] for r in recent], [r[2] for r in recent])
        cands.append(a2 + b2 * xt)
    cands.append(g[-1])
    cands.sort()
    g_t = cands[len(cands) // 2]
    spread = cands[-1] - cands[0]
    extrap = max(0.0, xt - xs[-1])
    sigma = 0.14 + 0.45 * spread + 0.85 * rmse * math.sqrt(extrap)
    return g_t, min(0.85, sigma)


def slope_fit(g, ys_i, ws, class_mean):
    """One trophy's slope on the index, shrunk toward its class mean.

    Returns (a, b, resid_sigma) for logit(rate) ~ a + b * g.
    """
    pts = [(g[j], ys_i[j], ws[j]) for j in range(len(g)) if ys_i[j] is not None]
    distinct = len({round(y, 3) for _x, y, _w in pts})
    if len(pts) < 3 or distinct < 2:
        if pts:
            a = sum(w * (y - class_mean * x) for x, y, w in pts) / sum(w for _x, _y, w in pts)
        else:
            a = 0.0
        return a, class_mean, 0.35
    xs = [p[0] for p in pts]
    ys_ = [p[1] for p in pts]
    w_ = [p[2] for p in pts]
    a, b, rmse = wls_line(xs, ys_, w_)
    kappa = 6.0
    n_eff = len(pts)
    b = (n_eff * b + kappa * class_mean) / (n_eff + kappa)
    a = sum(w * (y - b * x) for x, y, w in pts) / sum(w_)
    resid = math.sqrt(sum(w * (y - a - b * x) ** 2 for x, y, w in pts) / sum(w_))
    return a, b, min(0.9, max(0.12, resid))


def index_model(states, u_of, u_target, trophies, classes, active):
    """Per-trophy (median_logit, sigma) from the index model."""
    g, ws, ys = two_way(states, u_of, active)
    g_t, sig_g = index_trend(states, u_of, g, ws, u_target)
    raw = {}
    for i in active:
        raw[i] = slope_fit(g, ys[i], ws, 1.0)
    class_b = {}
    for cls in {classes[i][0] for i in active}:
        bs = [raw[i][1] for i in active if classes[i][0] == cls]
        if bs:
            mean_b = sum(bs) / len(bs)
            class_b[cls] = (len(bs) * mean_b + 12.0 * 1.0) / (len(bs) + 12.0)
    out = {}
    for i in active:
        a, b, resid = slope_fit(g, ys[i], ws, class_b.get(classes[i][0], 1.0))
        med = a + b * g_t
        sig = math.sqrt((b * sig_g) ** 2 + (0.7 * resid) ** 2 + 0.08 ** 2)
        out[i] = (med, sig)
    return out


# --- the earner model --------------------------------------------------------

def fit_players(states, u_of, u_target):
    """Forecast players tracked at u_target; returns (median, sigma_ln).

    An ensemble of a saturating exponential, a log-linear trend and a damped
    power law (growth factor capped), combined by median; the spread is the
    models' disagreement plus fit error, grown by the extrapolation.
    """
    pts = [(u_of(t), n) for (t, n, _r) in states if n and n > 0]
    if not pts:
        return 1000.0, 0.5
    if len(pts) == 1:
        return float(pts[0][1]), 0.5
    us = [p[0] for p in pts]
    ns = [float(p[1]) for p in pts]
    lnu = [math.log(u) for u in us]
    lnn = [math.log(n) for n in ns]
    ws = [1.0] * len(pts)
    preds = []
    a, b, rmse = wls_line(lnu, ns, ws)
    preds.append(max(1.0, a + b * math.log(u_target)))
    a2, b2, _r = wls_line(lnu, lnn, ws)
    growth = math.exp(b2 * (math.log(u_target) - lnu[-1]))
    growth = min(3.5, max(1.0, growth))
    preds.append(ns[-1] * growth)
    best = None
    for tau in (10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0):
        gs = [1.0 - math.exp(-u / tau) for u in us]
        den = sum(g * g for g in gs)
        if den <= 1e-9:
            continue
        ninf = sum(g * n for g, n in zip(gs, ns)) / den
        sse = sum((n - ninf * g) ** 2 for g, n in zip(gs, ns))
        if best is None or sse < best[0]:
            best = (sse, ninf * (1.0 - math.exp(-u_target / tau)))
    if best is not None:
        preds.append(max(1.0, best[1]))
    preds = [max(p, ns[-1]) for p in preds]
    preds.sort()
    median = preds[len(preds) // 2]
    spread = (math.log(preds[-1]) - math.log(preds[0])) / 2.0
    extrap = max(0.0, math.log(u_target / us[-1]))
    sigma = math.sqrt(spread ** 2 + rmse ** 2 / max(1.0, ns[-1])) * (1.0 + 0.5 * extrap)
    return median, min(0.5, max(0.05, sigma))


SIGMA_GRID = (0.55, 0.7, 0.85, 1.0, 1.2)


def fit_one_trophy(us, es, mu_prior, k_prior, sigma, u_target):
    """Fit E(u) = K * Phi((ln u - mu)/sigma) for one trophy, regularised.

    K is linear given mu, so mu is gridded then refined. Two pseudo-
    observations of the prior's own curve (at the last state and at the
    target, each worth about three states) keep a fit made on the curve's
    foot from collapsing to a small ceiling with an early midpoint.
    """
    lnu = [math.log(u) for u in us]
    ws = [1.0 + 0.5 * j / max(1, len(us) - 1) for j in range(len(us))]
    obs = list(zip(lnu, es, ws))
    for uu in (us[-1], u_target):
        e_p = k_prior * phi((math.log(uu) - mu_prior) / sigma)
        obs.append((math.log(uu), e_p, 3.0))

    def solve_k(mu):
        num = 0.0
        den = 0.0
        for x, e, w in obs:
            s = phi((x - mu) / sigma)
            num += w * s * e / (e + 8.0)
            den += w * s * s / (e + 8.0)
        return num / den if den > 1e-12 else k_prior

    def sse(mu, k):
        tot = 0.0
        for x, e, w in obs:
            d = e - k * phi((x - mu) / sigma)
            tot += w * d * d / (e + 8.0)
        return tot

    lo = lnu[0] - 0.6
    hi = math.log(u_target) + 1.2
    best_mu, best_k, best_s = mu_prior, k_prior, None
    steps = 46
    for i in range(steps + 1):
        mu = lo + (hi - lo) * i / steps
        k = solve_k(mu)
        s = sse(mu, k)
        if best_s is None or s < best_s:
            best_mu, best_k, best_s = mu, k, s
    step = (hi - lo) / steps
    a, b = best_mu - step, best_mu + step
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    dd = a + gr * (b - a)
    fc = sse(c, solve_k(c))
    fd = sse(dd, solve_k(dd))
    for _ in range(24):
        if fc < fd:
            b, dd, fd = dd, c, fc
            c = b - gr * (b - a)
            fc = sse(c, solve_k(c))
        else:
            a, c, fc = c, dd, fd
            dd = a + gr * (b - a)
            fd = sse(dd, solve_k(dd))
    mu = (a + b) / 2.0
    k = solve_k(mu)
    if sse(mu, k) < best_s:
        best_mu, best_k = mu, k
    resid = []
    for x, e in zip(lnu, es):
        pred = best_k * phi((x - best_mu) / sigma)
        resid.append(abs(math.log(max(e, 0.35)) - math.log(max(pred, 0.35))))
    rmse = math.sqrt(sum(r * r for r in resid) / len(resid)) if resid else 0.3
    return best_k, best_mu, rmse


def earner_model(states, u_of, u_target, trophies, priors):
    """Per-trophy (median_logit, sigma) from the earner decomposition."""
    m = len(states)
    us = [u_of(t) for (t, _n, _r) in states]
    n_med, n_sig = fit_players(states, u_of, u_target)
    series = []
    for i in range(len(trophies)):
        es = []
        for (_t, n, rates) in states:
            r = rates[i]
            es.append(None if r is None else r * n / 100.0)
        series.append(es)

    def usable(i):
        pts = [(us[j], series[i][j]) for j in range(m) if series[i][j] is not None]
        distinct = len({round(e, 3) for _u, e in pts})
        return pts, distinct

    best_sigma, best_total = 0.85, None
    fits_by_sigma = {}
    for sigma in SIGMA_GRID:
        total = 0.0
        fits = {}
        for i in range(len(trophies)):
            pts, distinct = usable(i)
            if len(pts) >= 3 and distinct >= 2:
                mu_p, share_p = priors[i]
                k, mu, rmse = fit_one_trophy([p[0] for p in pts], [p[1] for p in pts],
                                              mu_p, share_p * n_med, sigma, u_target)
                fits[i] = (k, mu, rmse)
                total += rmse
        fits_by_sigma[sigma] = fits
        if best_total is None or total < best_total:
            best_sigma, best_total = sigma, total
    fits = fits_by_sigma[best_sigma]

    out = {}
    ln_ut = math.log(u_target)
    for i in range(len(trophies)):
        mu_p, share_p = priors[i]
        k_prior = share_p * n_med
        pts, _distinct = usable(i)
        e_last = pts[-1][1] if pts else 0.0
        if i in fits:
            k, mu, rmse = fits[i]
            sig = math.sqrt(0.30 ** 2 + (0.8 * rmse) ** 2 + n_sig ** 2)
        else:
            mu, k = mu_p, (k_prior * 0.7 if e_last <= 0 else k_prior)
            sig = math.sqrt(0.45 ** 2 + n_sig ** 2)
        s_last = phi((math.log(us[-1]) - mu) / best_sigma)
        if s_last > 0.02:
            k = max(k, e_last / s_last)
        k = min(k, 1.05 * n_med)
        e_t = k * phi((ln_ut - mu) / best_sigma)
        rate = 100.0 * e_t / n_med
        out[i] = (logit(rate), min(1.0, sig))
    return out


# ------------------------------------------------------------- stale fallback

# Per-hour earner growth by class while a launch page sits stale: the tracked
# base is broadening much faster than any trophy's earners accumulate.
HOURLY_GROWTH = {
    "chapter": 1.085, "finale": 1.070, "mission": 1.078, "misc": 1.065,
    "grind": 1.040, "completion": 1.060, "platinum": 1.055,
}


def stale_forecast(case, states, classes):
    """Forecast when the page never changed during the history.

    The visible state is at least as old as the recording, and at launch the
    next refresh multiplies the tracked base (the practice's first refresh
    multiplied it sixfold while earners of the ordinary trophies rose by a
    third). Median: earners grow at the class rate, players at the launch
    rate, and the ratio is the rate. The spread prices the unknown size of
    the jump.
    """
    first_t = states[0][0]
    last = states[-1]
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    near_launch = (parse_time(case["at"]) - release).total_seconds() <= 72 * 3600
    window_h = max(0.5, (parse_time(case["target"]) - first_t).total_seconds() / 3600.0)
    if near_launch:
        g_players = min(14.0, math.exp(0.85 * math.sqrt(window_h)))
    else:
        g_players = min(3.0, math.exp(0.30 * math.sqrt(window_h)))
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last[2][i]
        if rate is None:
            rate = 0.0
        if rate >= 99.9:
            rows.append([99.99, 100.0, 100.0, 100.0, 100.0])
            continue
        cls = classes[i][0]
        g_earn = HOURLY_GROWTH[cls] ** window_h
        med = rate * g_earn / g_players
        med = max(med, rate * 0.03)  # earners grow; rates cannot fall forever
        vals = []
        for dln in (-0.95, -0.42, 0.0, 0.42, 0.95):
            lg = logit(med) + dln * 1.05
            vals.append(inv_logit(lg))
        vals.sort()
        rows.append([min(100.0, max(0.0, v)) for v in vals])
    return rows


# ------------------------------------------------------------- main forecast

def pinned(trophy_states_rates):
    """True when a trophy sat at 100% in every state: its earners are the
    whole tracked base, in force for the whole history, so it stays there."""
    vals = [r for r in trophy_states_rates if r is not None]
    return bool(vals) and all(v >= 99.9 for v in vals)


def forecast_rows(case):
    trophies = case["trophies"]
    history = case["history"]
    first_t = parse_time(history[0]["slot"])
    target_dt = parse_time(case["target"])

    def u_of(t):
        return 24.0 + (t - first_t).total_seconds() / 3600.0

    u_target = u_of(target_dt)
    states = compress_states(history)
    classes = [classify(t) for t in trophies]
    priors = [prior_for(cls, ch) for cls, ch in classes]

    n_distinct = len({n for (_t, n, _r) in states})
    if len(states) <= 1 or n_distinct <= 1:
        return stale_forecast(case, states, classes)

    last_rates = states[-1][2]
    is_pinned = [pinned([rates[i] for (_t, _n, rates) in states]) for i in range(len(trophies))]
    active = [i for i in range(len(trophies))
              if last_rates[i] is not None and not is_pinned[i]]
    idx = index_model(states, u_of, u_target, trophies, classes, active)
    ern = earner_model(states, u_of, u_target, trophies, priors)

    w_idx = 0.45
    # With few distinct states the extrapolations are built on sand: shrink
    # every median toward the last observed rate, the shrinkage fading as
    # the state count grows. Cheap insurance against a wild fit.
    gamma = min(0.5, max(0.06, 0.55 - 0.045 * (len(states) - 2)))
    rows = []
    for i, trophy in enumerate(trophies):
        if is_pinned[i]:
            rows.append([99.99, 100.0, 100.0, 100.0, 100.0])
            continue
        if i in idx:
            m1, s1 = idx[i]
            m2, s2 = ern[i]
            med = w_idx * m1 + (1.0 - w_idx) * m2
            sig = 0.85 * math.sqrt(w_idx * s1 * s1 + (1.0 - w_idx) * s2 * s2
                                   + 0.35 * abs(m1 - m2))
        else:
            med, sig = ern[i]
        if last_rates[i] is not None:
            med = (1.0 - gamma) * med + gamma * logit(last_rates[i])
        rows.append([inv_logit(med + z * sig) for z in Z_OF_LEVEL])
    apply_caps(rows, trophies, classes)
    return rows


def apply_caps(rows, trophies, classes):
    """Structural caps: nothing outruns the trophies it presupposes."""
    others_min = []
    for lvl in range(5):
        others_min.append(min(rows[j][lvl] for j in range(len(rows))
                              if trophies[j]["tier"] != "platinum"
                              and classes[j][0] != "completion"))
    for j, t in enumerate(trophies):
        if t["tier"] == "platinum" or classes[j][0] == "completion":
            for lvl in range(5):
                rows[j][lvl] = min(rows[j][lvl], others_min[lvl])
    # Story chapters in order, then the finale: later steps cap at earlier ones.
    chain = []
    for j, t in enumerate(trophies):
        cls, ch = classes[j]
        if cls == "chapter":
            chain.append((ch, j))
    chain.sort()
    order = [j for _ch, j in chain]
    order += [j for j in range(len(trophies)) if classes[j][0] == "finale"]
    for a, b in zip(order, order[1:]):
        for lvl in range(5):
            rows[b][lvl] = min(rows[b][lvl], rows[a][lvl])
    for row in rows:
        row.sort()


def forecast_case(case):
    rows = forecast_rows(case)
    return {"trophies": [{"id": t["id"], "q": [round4(v) for v in rows[j]]}
                         for j, t in enumerate(case["trophies"])]}


def emergency_rows(case):
    """Wide persistence rows: a valid forecast whatever went wrong above."""
    last = case["history"][-1]
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        r = last["rates"][i] or 0.0
        lg = logit(r)
        rows.append({"id": trophy["id"],
                     "q": [round4(inv_logit(lg + z * 1.2)) for z in Z_OF_LEVEL]})
    return {"trophies": rows}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(IN_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            out = forecast_case(case)
        except Exception:
            out = emergency_rows(case)
        (OUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
