# -*- coding: utf-8 -*-


from __future__ import annotations
import math
import json
import dataclasses as dc
from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import digamma, polygamma
from pathlib import Path

DATA_PATH = Path("/content/ndt_tests_corrigido.csv")  # <-- set after uploading
TIME_COL = "timestamp"
COLUMN_MAP = {
    "client": "client",
    "server": "server",
    "dl_bps": "download_throughput_bps",
    "ul_bps": "upload_throughput_bps",
    "dl_rtt": "rtt_download_sec",
    "ul_rtt": "rtt_upload_sec",
    "loss_frac": "loss_fraction",
}


Q_LIST = [0.5, 0.9, 0.95, 0.99]


N_PACKETS = 1000

RNG = np.random.default_rng(12345)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    required_src = [
        "download_throughput_bps",
        "upload_throughput_bps",
        "rtt_download_sec",
        "rtt_upload_sec",
        "client",
        "server",
    ]
    for v in required_src:
        if v not in df.columns:
            raise KeyError(f"Missing expected column '{v}'.")

    if "packet_loss_percent" in df.columns:
        df["loss_fraction"] = pd.to_numeric(df["packet_loss_percent"], errors="coerce") / 100.0
    elif "loss_fraction" in df.columns:
        df["loss_fraction"] = pd.to_numeric(df["loss_fraction"], errors="coerce")
    else:
        df["loss_fraction"] = np.nan

    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in [COLUMN_MAP["dl_bps"], COLUMN_MAP["ul_bps"]]:
        df = df[df[c] > 0]
    for c in [COLUMN_MAP["dl_rtt"], COLUMN_MAP["ul_rtt"]]:
        df = df[df[c] > 0]
    lf = COLUMN_MAP["loss_frac"]
    df = df[(df[lf] >= 0) & (df[lf] <= 1)]
    df = df.drop_duplicates()
    return df


def temporal_train_test_split(df: pd.DataFrame, train_frac: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if TIME_COL not in df.columns:
        n = len(df)
        cut = int(math.floor(train_frac * n))
        return df.iloc[:cut].copy(), df.iloc[cut:].copy()
    dsort = df.sort_values(TIME_COL)
    n = len(dsort)
    cut = int(math.floor(train_frac * n))
    return dsort.iloc[:cut].copy(), dsort.iloc[cut:].copy()

def describe_grouped(
    df: pd.DataFrame,
    by: List[str],
    vars_: List[str],
    quantiles: List[float] = Q_LIST,
) -> pd.DataFrame:
    def qfunc(x: pd.Series) -> pd.Series:
        out = {
            "mean": x.mean(),
            "median": x.median(),
            "var": x.var(ddof=1),
            "std": x.std(ddof=1),
        }
        for q in quantiles:
            out[f"q{int(100*q)}"] = x.quantile(q)
        return pd.Series(out)

    pieces = []
    for v in vars_:
        g = df.groupby(by)[v].apply(qfunc).unstack(-1)
        g["variable"] = v
        pieces.append(g.reset_index())
    out = pd.concat(pieces, axis=0, ignore_index=True)
    cols = by + ["variable", "mean", "median", "var", "std"] + [f"q{int(100*q)}" for q in quantiles]
    return out[cols]


def plot_hist_box_scatter(
    df: pd.DataFrame,
    client: Optional[str] = None,
    server: Optional[str] = None,
    pair_for_scatter: Tuple[str, str] = (COLUMN_MAP["dl_bps"], COLUMN_MAP["dl_rtt"]),
    bins: int = 40,
):
    dat = df.copy()
    if client is not None:
        dat = dat[dat[COLUMN_MAP["client"]] == client]
    if server is not None:
        dat = dat[dat[COLUMN_MAP["server"]] == server]

    vars_ = [COLUMN_MAP["dl_bps"], COLUMN_MAP["ul_bps"], COLUMN_MAP["dl_rtt"], COLUMN_MAP["ul_rtt"], COLUMN_MAP["loss_frac"]]

    for v in vars_:
        plt.figure(figsize=(6,4))
        plt.hist(dat[v].dropna().values, bins=bins, density=True)
        plt.title(f"Histogram – {v}")
        plt.xlabel(v); plt.ylabel("Density"); plt.show()

        plt.figure(figsize=(4,5))
        plt.boxplot(dat[v].dropna().values, vert=True, showfliers=True)
        plt.title(f"Boxplot – {v}")
        plt.ylabel(v); plt.show()

    x, y = pair_for_scatter
    plt.figure(figsize=(6,4))
    plt.scatter(dat[x].values, dat[y].values, s=10, alpha=0.5)
    plt.xlabel(x); plt.ylabel(y)
    plt.title(f"Scatter – {x} vs {y}")
    plt.show()

def mle_normal(x: np.ndarray) -> Tuple[float, float]:
    """Return (mu_hat, sigma2_hat) MLE for Normal IID."""
    x = np.asarray(x, dtype=float)
    mu_hat = x.mean()
    sigma2_hat = ((x - mu_hat) ** 2).mean()
    return mu_hat, sigma2_hat


def mle_gamma_k_beta(x: np.ndarray) -> Tuple[float, float]:
    """Return (k_hat, beta_hat) MLE for Gamma(k, beta) with PDF beta^k / Gamma(k) * x^{k-1} e^{-beta x}.
    Uses Newton iterations on k, with beta = k / mean(x). Robust to edge cases.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 2:
        m = float(x.mean()) if x.size else 1.0
        k = 1.0
        beta = k / m if m > 0 else 1.0
        return float(k), float(beta)

    m = float(x.mean())
    s2 = float(x.var(ddof=0))
    k = (m * m / s2) if s2 > 0 else 1.0
    k = max(k, 1e-6)

    mean_logx = float(np.log(x).mean())


    for _ in range(50):
        psi = digamma(k)
        psi1 = polygamma(1, k)
        g = math.log(k) - psi - math.log(m) + mean_logx
        gprime = (1.0 / k) - psi1
        step = g / gprime
        k_new = k - step
        if not np.isfinite(k_new) or k_new <= 0:
            k_new = k / 2.0
        if abs(k_new - k) < 1e-10:
            k = k_new
            break
        k = k_new

    beta = k / m if m > 0 else 1.0
    return float(k), float(beta)

def overlay_hist_with_pdf(x: np.ndarray, pdf, label: str, bins: int = 40):
    x = np.asarray(x, dtype=float)
    plt.figure(figsize=(6,4))
    plt.hist(x, bins=bins, density=True, alpha=0.5)
    xs = np.linspace(x.min(), x.max(), 400)
    plt.plot(xs, pdf(xs), lw=2, label=label)
    plt.legend(); plt.title("Data vs fitted PDF"); plt.show()


def qqplot_vs_dist(x: np.ndarray, dist: stats.rv_continuous, params: Tuple, title: str):
    q = np.linspace(0.01, 0.99, 99)
    th = dist.ppf(q, *params)
    samp = np.quantile(x, q)
    plt.figure(figsize=(5,5))
    plt.scatter(th, samp, s=12)
    lims = [min(th.min(), samp.min()), max(th.max(), samp.max())]
    plt.plot(lims, lims, ls='--')
    plt.xlabel("Theoretical quantiles"); plt.ylabel("Sample quantiles")
    plt.title(title)
    plt.show()


def normal_normal_posterior(mu0: float, tau0_sq: float, sigma_sq: float, xbar: float, n: int) -> Tuple[float, float]:
    prec0 = 1.0 / tau0_sq
    prec_data = n / sigma_sq
    tau_n_sq = 1.0 / (prec0 + prec_data)
    mu_n = tau_n_sq * (prec0 * mu0 + prec_data * xbar)
    return mu_n, tau_n_sq


def normal_predictive(mu_n: float, tau_n_sq: float, sigma_sq: float) -> Tuple[float, float]:
    return mu_n, sigma_sq + tau_n_sq


def beta_binomial_update(a0: float, b0: float, x_tot: int, n_tot: int) -> Tuple[float, float]:
    a_n = a0 + x_tot
    b_n = b0 + (n_tot - x_tot)
    return a_n, b_n


def beta_binomial_predictive_mean_var(a_n: float, b_n: float, n_star: int) -> Tuple[float, float]:
    mean = n_star * (a_n / (a_n + b_n))
    var = n_star * (a_n * b_n * (a_n + b_n + n_star)) / ((a_n + b_n) ** 2 * (a_n + b_n + 1))
    return mean, var


def gamma_gamma_posterior(a0: float, b0: float, k_fixed: float, y: np.ndarray) -> Tuple[float, float]:
    n = len(y)
    a_n = a0 + n * k_fixed
    b_n = b0 + float(np.sum(y))
    return a_n, b_n


def gamma_predictive_mean_var(a_n: float, b_n: float, k_fixed: float) -> Tuple[Optional[float], Optional[float]]:
    mean = None
    var = None
    if a_n > 1:
        mean = k_fixed * (b_n / (a_n - 1))
    if a_n > 2:
        var = k_fixed * (k_fixed + a_n - 1) * (b_n ** 2) / ((a_n - 1) ** 2 * (a_n - 2))
    return mean, var


def run_eda(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    by_client = describe_grouped(df, [COLUMN_MAP["client"]], [COLUMN_MAP["dl_bps"], COLUMN_MAP["ul_bps"], COLUMN_MAP["dl_rtt"], COLUMN_MAP["ul_rtt"], COLUMN_MAP["loss_frac"]])
    by_server = describe_grouped(df, [COLUMN_MAP["server"]], [COLUMN_MAP["dl_bps"], COLUMN_MAP["ul_bps"], COLUMN_MAP["dl_rtt"], COLUMN_MAP["ul_rtt"], COLUMN_MAP["loss_frac"]])
    return {"by_client": by_client, "by_server": by_server}


def select_two_entities(df: pd.DataFrame) -> Tuple[str, str]:
    g = df.groupby(COLUMN_MAP["client"])[COLUMN_MAP["dl_rtt"]].median().sort_values()
    if len(g) >= 2:
        return g.index[0], g.index[-1]
    uniq = df[COLUMN_MAP["client"]].dropna().unique()
    return (uniq[0], uniq[1]) if len(uniq) >= 2 else (uniq[0], uniq[0])


def fit_mle_blocks(train: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out = {}
    for key in ("dl_rtt", "ul_rtt"):
        x = train[COLUMN_MAP[key]].dropna().to_numpy()
        mu_hat, s2_hat = mle_normal(x)
        out[key] = {"mu_hat": mu_hat, "sigma2_hat": s2_hat}
    for key in ("dl_bps", "ul_bps"):
        y = train[COLUMN_MAP[key]].dropna().to_numpy()
        k_hat, beta_hat = mle_gamma_k_beta(y)
        out[key] = {"k_hat": k_hat, "beta_hat": beta_hat}
    lf = train[COLUMN_MAP["loss_frac"]].dropna().to_numpy()
    x_counts = np.rint(lf * N_PACKETS).astype(int)
    out["loss"] = {
        "x_tot": int(x_counts.sum()),
        "n_tot": int(N_PACKETS * len(x_counts)),
    }
    return out


def run_bayes_blocks(train: pd.DataFrame, mle: Dict[str, Dict[str, float]], priors: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out = {}
    for key in ("dl_rtt", "ul_rtt"):
        mu0 = priors[key]["mu0"]; tau0_sq = priors[key]["tau0_sq"]
        sigma_sq = mle[key]["sigma2_hat"]
        x = train[COLUMN_MAP[key]].dropna().to_numpy()
        mu_n, tau_n_sq = normal_normal_posterior(mu0, tau0_sq, sigma_sq, x.mean(), len(x))
        pred_mu, pred_var = normal_predictive(mu_n, tau_n_sq, sigma_sq)
        out[key] = {"mu_n": mu_n, "tau_n_sq": tau_n_sq, "pred_mean": pred_mu, "pred_var": pred_var}
    for key in ("dl_bps", "ul_bps"):
        a0 = priors[key]["a0"]; b0 = priors[key]["b0"]
        k_fixed = mle[key]["k_hat"]
        y = train[COLUMN_MAP[key]].dropna().to_numpy()
        a_n, b_n = gamma_gamma_posterior(a0, b0, k_fixed, y)
        pred_mean, pred_var = gamma_predictive_mean_var(a_n, b_n, k_fixed)
        out[key] = {"a_n": a_n, "b_n": b_n, "k_fixed": k_fixed, "pred_mean": pred_mean, "pred_var": pred_var}
    a0 = priors["loss"]["a0"]; b0 = priors["loss"]["b0"]
    lf = train[COLUMN_MAP["loss_frac"]].dropna().to_numpy()
    x_counts = np.rint(lf * N_PACKETS).astype(int)
    x_tot = int(x_counts.sum()); n_tot = int(N_PACKETS * len(x_counts))
    a_n, b_n = beta_binomial_update(a0, b0, x_tot, n_tot)
    pred_mean, pred_var = beta_binomial_predictive_mean_var(a_n, b_n, N_PACKETS)
    out["loss"] = {"a_n": a_n, "b_n": b_n, "pred_mean_counts": pred_mean, "pred_var_counts": pred_var, "pred_mean_frac": pred_mean / N_PACKETS}
    return out


def evaluate_on_test(test: pd.DataFrame, bayes_pred: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for key in ("dl_rtt", "ul_rtt"):
        x = test[COLUMN_MAP[key]].dropna().to_numpy()
        rows.append({
            "variable": key,
            "test_mean": float(np.mean(x)),
            "test_var": float(np.var(x, ddof=1)),
            "pred_mean": float(bayes_pred[key]["pred_mean"]),
            "pred_var": float(bayes_pred[key]["pred_var"]),
        })
    for key in ("dl_bps", "ul_bps"):
        y = test[COLUMN_MAP[key]].dropna().to_numpy()
        rows.append({
            "variable": key,
            "test_mean": float(np.mean(y)),
            "test_var": float(np.var(y, ddof=1)),
            "pred_mean": float(bayes_pred[key]["pred_mean"]) if bayes_pred[key]["pred_mean"] is not None else np.nan,
            "pred_var": float(bayes_pred[key]["pred_var"]) if bayes_pred[key]["pred_var"] is not None else np.nan,
        })
    lf = test[COLUMN_MAP["loss_frac"]].dropna().to_numpy()
    rows.append({
        "variable": "loss_frac",
        "test_mean": float(np.mean(lf)),
        "test_var": float(np.var(lf, ddof=1)),
        "pred_mean": float(bayes_pred["loss"]["pred_mean_frac"]),
        "pred_var": float(bayes_pred["loss"]["pred_var_counts"]) / (N_PACKETS**2),
    })
    return pd.DataFrame(rows)

DEFAULT_PRIORS = {
    "dl_rtt": {"mu0": 0.05, "tau0_sq": 1.0},
    "ul_rtt": {"mu0": 0.05, "tau0_sq": 1.0},
    "dl_bps": {"a0": 1.0, "b0": 1.0},
    "ul_bps": {"a0": 1.0, "b0": 1.0},
    "loss": {"a0": 1.0, "b0": 1.0},
}

if __name__ == "__main__":
    df0 = load_data(DATA_PATH)
    df = basic_clean(df0)

    eda_tables = run_eda(df)
    print("\n>>> EDA by client (head):\n", eda_tables["by_client"].head())
    print("\n>>> EDA by server (head):\n", eda_tables["by_server"].head())

    c_low, c_high = select_two_entities(df)
    print(f"Selected contrasting clients: {c_low} vs {c_high}")

    train, test = temporal_train_test_split(df, train_frac=0.7)

    mle = fit_mle_blocks(train)
    print("\n>>> MLE summaries:\n", json.dumps(mle, indent=2))

    priors = DEFAULT_PRIORS.copy()
    for key in ("dl_rtt", "ul_rtt"):
        priors[key] = dict(priors[key])
        priors[key]["mu0"] = float(train[COLUMN_MAP[key]].mean())
        priors[key]["tau0_sq"] = float(train[COLUMN_MAP[key]].var(ddof=1))  # weakly informative

    bayes_pred = run_bayes_blocks(train, mle, priors)
    print("\n>>> Bayes posterior/predictive (summaries):\n", json.dumps(bayes_pred, indent=2))

    comp = evaluate_on_test(test, bayes_pred)
    print("\n>>> Predictive vs Test (means/vars):\n", comp)


    pass

client 1 and client 5


def gamma_log_likelihood(x: np.ndarray, k: float, beta: float) -> float:


    x_positive = x[np.isfinite(x) & (x > 0)]
    if x_positive.size == 0:
        return -np.inf 
    return float(np.sum(stats.gamma.logpdf(x_positive, a=k, scale=1/beta)))

print("Defined gamma_log_likelihood function.")


dl_bps_client01 = train[train[COLUMN_MAP["client"]] == "client01"][COLUMN_MAP["dl_bps"]]
dl_bps_client05 = train[train[COLUMN_MAP["client"]] == "client05"][COLUMN_MAP["dl_bps"]]

print(f"Number of dl_bps measurements for client01: {len(dl_bps_client01)}")
print(f"Number of dl_bps measurements for client05: {len(dl_bps_client05)}")

k1_hat, beta1_hat = mle_gamma_k_beta(dl_bps_client01.to_numpy())
k5_hat, beta5_hat = mle_gamma_k_beta(dl_bps_client05.to_numpy())

print(f"MLE for client01 (dl_bps): k_hat = {k1_hat:.4f}, beta_hat = {beta1_hat:.10e}")
print(f"MLE for client05 (dl_bps): k_hat = {k5_hat:.4f}, beta_hat = {beta5_hat:.10e}")



log_likelihood_h1_client01 = gamma_log_likelihood(dl_bps_client01.to_numpy(), k1_hat, beta1_hat)
log_likelihood_h1_client05 = gamma_log_likelihood(dl_bps_client05.to_numpy(), k5_hat, beta5_hat)

# Sum of log-likelihoods for H1
log_likelihood_h1 = log_likelihood_h1_client01 + log_likelihood_h1_client05

print(f"Log-likelihood for client01 (H1): {log_likelihood_h1_client01:.2f}")
print(f"Log-likelihood for client05 (H1): {log_likelihood_h1_client05:.2f}")
print(f"Total Log-likelihood for H1 (unequal rates): {log_likelihood_h1:.2f}")

dl_bps_combined = pd.concat([dl_bps_client01, dl_bps_client05])
k0_hat, beta0_hat = mle_gamma_k_beta(dl_bps_combined.to_numpy())

print(f"MLE for combined clients (dl_bps): k_hat = {k0_hat:.4f}, beta_hat = {beta0_hat:.10e}")


log_likelihood_h0 = gamma_log_likelihood(dl_bps_combined.to_numpy(), k0_hat, beta0_hat)

print(f"Log-likelihood for H0 (equal rates): {log_likelihood_h0:.2f}")


lrt_statistic = 2 * (log_likelihood_h1 - log_likelihood_h0)
df_lrt = 2 # (k1, beta1, k5, beta5) - (k0, beta0) = 4 - 2 = 2
p_value = stats.chi2.sf(lrt_statistic, df_lrt)

print(f"LRT Statistic: {lrt_statistic:.2f}")
print(f"Degrees of Freedom: {df_lrt}")
print(f"P-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print(f"At alpha = {alpha}, we reject the null hypothesis. There is a significant difference between the dl_bps rates of client01 and client05.")
else:
    print(f"At alpha = {alpha}, we fail to reject the null hypothesis. There is no significant difference between the dl_bps rates of client01 and client05.")


def normal_log_likelihood(x: np.ndarray, mu: float, sigma: float) -> float:
    """Calculates the log-likelihood of data for a Normal distribution.

    Args:
        x (np.ndarray): The data points.
        mu (float): The mean of the Normal distribution.
        sigma (float): The standard deviation of the Normal distribution.

    Returns:
        float: The total log-likelihood.
    """
    x_finite = x[np.isfinite(x)]
    if x_finite.size == 0:
        return -np.inf # No valid data points, log-likelihood is negative infinity
    return float(np.sum(stats.norm.logpdf(x_finite, loc=mu, scale=sigma)))

print("Defined normal_log_likelihood function.")


dl_rtt_client01 = train[train[COLUMN_MAP["client"]] == "client01"][COLUMN_MAP["dl_rtt"]]
dl_rtt_client05 = train[train[COLUMN_MAP["client"]] == "client05"][COLUMN_MAP["dl_rtt"]]

print(f"Number of dl_rtt measurements for client01: {len(dl_rtt_client01)}")
print(f"Number of dl_rtt measurements for client05: {len(dl_rtt_client05)}")


mu1_hat, sigma2_1_hat = mle_normal(dl_rtt_client01.to_numpy())
sigma1_hat = np.sqrt(sigma2_1_hat)

mu5_hat, sigma2_5_hat = mle_normal(dl_rtt_client05.to_numpy())
sigma5_hat = np.sqrt(sigma2_5_hat)

print(f"MLE for client01 (dl_rtt): mu_hat = {mu1_hat:.6f}, sigma_hat = {sigma1_hat:.6f}")
print(f"MLE for client05 (dl_rtt): mu_hat = {mu5_hat:.6f}, sigma_hat = {sigma5_hat:.6f}")



log_likelihood_h1_client01_rtt = normal_log_likelihood(dl_rtt_client01.to_numpy(), mu1_hat, sigma1_hat)
log_likelihood_h1_client05_rtt = normal_log_likelihood(dl_rtt_client05.to_numpy(), mu5_hat, sigma5_hat)

# Sum of log-likelihoods for H1
log_likelihood_h1_rtt = log_likelihood_h1_client01_rtt + log_likelihood_h1_client05_rtt

print(f"Log-likelihood for client01 dl_rtt (H1): {log_likelihood_h1_client01_rtt:.2f}")
print(f"Log-likelihood for client05 dl_rtt (H1): {log_likelihood_h1_client05_rtt:.2f}")
print(f"Total Log-likelihood for H1 (unequal dl_rtt rates): {log_likelihood_h1_rtt:.2f}")


dl_rtt_combined = pd.concat([dl_rtt_client01, dl_rtt_client05])
mu0_rtt_hat, sigma2_0_rtt_hat = mle_normal(dl_rtt_combined.to_numpy())
sigma0_rtt_hat = np.sqrt(sigma2_0_rtt_hat)

print(f"MLE for combined clients (dl_rtt): mu_hat = {mu0_rtt_hat:.6f}, sigma_hat = {sigma0_rtt_hat:.6f}")


log_likelihood_h0_rtt = normal_log_likelihood(dl_rtt_combined.to_numpy(), mu0_rtt_hat, sigma0_rtt_hat)

print(f"Log-likelihood for H0 (equal dl_rtt rates): {log_likelihood_h0_rtt:.2f}")



lrt_statistic_rtt = 2 * (log_likelihood_h1_rtt - log_likelihood_h0_rtt)
df_lrt_rtt = 2 # (mu1, sigma1, mu5, sigma5) - (mu0, sigma0) = 4 - 2 = 2
p_value_rtt = stats.chi2.sf(lrt_statistic_rtt, df_lrt_rtt)

print(f"LRT Statistic for dl_rtt: {lrt_statistic_rtt:.2f}")
print(f"Degrees of Freedom for dl_rtt: {df_lrt_rtt}")
print(f"P-value for dl_rtt: {p_value_rtt:.4f}")

alpha = 0.05
if p_value_rtt < alpha:
    print(f"At alpha = {alpha}, we reject the null hypothesis for dl_rtt. There is a significant difference between the dl_rtt rates of client01 and client05.")
else:
    print(f"At alpha = {alpha}, we fail to reject the null hypothesis for dl_rtt. There is no significant difference between the dl_rtt rates of client01 and client05.")

