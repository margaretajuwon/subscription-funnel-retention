"""Subscription funnel, cohort retention and churn analysis.

All data is SYNTHETIC (generated below with a fixed seed). It mimics a digital subscription
business: visitors hit a paywall, some register, some subscribe, some later cancel.
Run: python analysis.py   -> writes results/*.png, results/*.csv and README.md
"""
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).parent
RES = OUT / "results"
RES.mkdir(exist_ok=True)
rng = np.random.default_rng(42)

# ---------------------------------------------------------------- 1. synthetic data
N = 60_000
channels = ["organic_search", "social", "email", "push", "direct"]
ch_p = [0.34, 0.20, 0.14, 0.12, 0.20]
# base rates differ by channel (this is the "signal" the analysis should recover)
reg_rate = {"organic_search": 0.14, "social": 0.07, "email": 0.26, "push": 0.18, "direct": 0.22}
sub_given_reg = {"organic_search": 0.20, "social": 0.12, "email": 0.34, "push": 0.22, "direct": 0.30}

visitors = pd.DataFrame({
    "visitor_id": np.arange(N),
    "channel": rng.choice(channels, N, p=ch_p),
    "first_seen": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 360, N), unit="D"),
})
visitors["hit_paywall"] = rng.random(N) < 0.55
visitors["registered"] = visitors.hit_paywall & (rng.random(N) < visitors.channel.map(reg_rate) * 1.6)
visitors["subscribed"] = visitors.registered & (rng.random(N) < visitors.channel.map(sub_given_reg))
subs = visitors[visitors.subscribed].copy()
# churn: monthly hazard higher early in life, lower for email/direct subscribers
base_h = {"organic_search": 0.075, "social": 0.11, "email": 0.05, "push": 0.08, "direct": 0.055}
subs["tenure_months"] = 0
alive = np.ones(len(subs), bool)
tenure = np.zeros(len(subs), int)
h = subs.channel.map(base_h).to_numpy()
for m in range(1, 13):
    hazard = h * (1.8 if m <= 2 else 1.0 if m <= 6 else 0.7)
    churn_now = alive & (rng.random(len(subs)) < hazard)
    tenure[alive & ~churn_now] = m
    alive &= ~churn_now
subs["tenure_months"] = tenure
subs["churned"] = ~alive
subs["sub_start"] = subs.first_seen + pd.to_timedelta(rng.integers(0, 14, len(subs)), unit="D")
subs["cohort_month"] = subs.sub_start.dt.to_period("M").astype(str)

# ---------------------------------------------------------------- 2. SQL funnel
con = sqlite3.connect(":memory:")
visitors.assign(first_seen=visitors.first_seen.astype(str)).to_sql("visitors", con, index=False)
funnel_sql = """
SELECT channel,
       COUNT(*)                                             AS visitors,
       SUM(hit_paywall)                                     AS paywall_views,
       SUM(registered)                                      AS registered,
       SUM(subscribed)                                      AS subscribed,
       ROUND(1.0*SUM(registered)/NULLIF(SUM(hit_paywall),0),4) AS paywall_to_reg,
       ROUND(1.0*SUM(subscribed)/NULLIF(SUM(registered),0),4) AS reg_to_sub,
       ROUND(1.0*SUM(subscribed)/NULLIF(SUM(hit_paywall),0),4) AS paywall_to_sub
FROM visitors GROUP BY channel ORDER BY paywall_to_sub DESC;
"""
funnel = pd.read_sql(funnel_sql, con)
funnel.to_csv(RES / "funnel_by_channel.csv", index=False)
tot = visitors[["hit_paywall", "registered", "subscribed"]].sum()
overall_drop = 1 - tot.registered / tot.hit_paywall

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(funnel.channel, funnel.paywall_to_sub * 100, color="#2b6cb0")
ax.set_ylabel("Paywall view -> subscription (%)")
ax.set_title("Funnel conversion by acquisition channel (synthetic data)")
plt.xticks(rotation=20)
plt.tight_layout(); plt.savefig(RES / "funnel_by_channel.png", dpi=140); plt.close()

# ---------------------------------------------------------------- 3. cohort retention
months = range(0, 13)
rows = []
for cohort, g in subs.groupby("cohort_month"):
    n0 = len(g)
    rows.append([cohort, n0] + [(g.tenure_months >= m).sum() / n0 for m in months])
ret = pd.DataFrame(rows, columns=["cohort", "size"] + [f"m{m}" for m in months]).sort_values("cohort")
ret = ret[ret["size"] >= 100]
ret.to_csv(RES / "cohort_retention.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 4.5))
im = ax.imshow(ret[[f"m{m}" for m in months]].to_numpy(), aspect="auto", cmap="Blues", vmin=0.3, vmax=1)
ax.set_yticks(range(len(ret))); ax.set_yticklabels(ret.cohort)
ax.set_xticks(range(13)); ax.set_xlabel("Months since subscription start"); ax.set_title("Cohort retention (synthetic data)")
fig.colorbar(im, label="share still subscribed"); plt.tight_layout()
plt.savefig(RES / "cohort_retention.png", dpi=140); plt.close()

# ---------------------------------------------------------------- 4. churn by tenure & channel
haz = []
for ch, g in subs.groupby("channel"):
    for m in range(1, 13):
        at_risk = (g.tenure_months >= m - 1).sum()
        churned_m = ((g.tenure_months == m - 1) & g.churned).sum()
        haz.append([ch, m, churned_m / at_risk if at_risk else np.nan])
haz = pd.DataFrame(haz, columns=["channel", "month", "hazard"])
fig, ax = plt.subplots(figsize=(7, 4))
for ch, g in haz.groupby("channel"):
    ax.plot(g.month, g.hazard * 100, marker="o", label=ch)
ax.set_xlabel("Month of subscription"); ax.set_ylabel("Monthly churn rate (%)"); ax.legend(fontsize=7)
ax.set_title("Churn hazard by tenure and channel (synthetic data)")
plt.tight_layout(); plt.savefig(RES / "churn_hazard.png", dpi=140); plt.close()

early = subs.assign(early_churn=(subs.tenure_months < 3) & subs.churned).groupby("channel").early_churn.mean().sort_values()
best, worst = funnel.iloc[0], funnel.iloc[-1]

# ---------------------------------------------------------------- 5. write README with real numbers
readme = f"""# Subscription funnel, cohort retention and churn analysis

**Synthetic data, real method.** The data in this project is generated by `analysis.py` with a fixed
random seed. It imitates a digital subscription business and contains no data from any employer.
Everything below is recomputed each time you run the script.

## Business questions
1. Where do visitors drop out between the paywall and a paid subscription, and does it differ by acquisition channel?
2. How does retention look by monthly cohort?
3. When in a subscriber's life do they churn, and which channels churn earliest?

## Approach
- SQL (SQLite) for the funnel: visitors -> paywall views -> registrations -> subscriptions, by channel.
- Cohort retention matrix in pandas (subscribers grouped by start month, share still active by month).
- Discrete-time churn hazard by tenure month and channel.

## Findings (from this synthetic dataset)
- Of {int(tot.hit_paywall):,} paywall views, {int(tot.subscribed):,} became subscribers
  ({tot.subscribed / tot.hit_paywall:.1%}). The largest single drop is paywall -> registration:
  {overall_drop:.0%} of paywall views never register.
- **{best.channel}** converts best from paywall to subscription ({best.paywall_to_sub:.1%});
  **{worst.channel}** converts worst ({worst.paywall_to_sub:.1%}).
- Early churn (cancelling within 3 months) is highest for **{early.index[-1]}** subscribers
  ({early.iloc[-1]:.1%}) and lowest for **{early.index[0]}** ({early.iloc[0]:.1%}).
- Churn is front-loaded: the hazard is highest in months 1-2 and falls after month 6, so onboarding
  and the first 60 days matter most for retention.

![funnel](results/funnel_by_channel.png)
![cohorts](results/cohort_retention.png)
![hazard](results/churn_hazard.png)

## What I would do with real data
- Validate tracking first (event definitions, duplicate or missing events) before trusting any funnel drop.
- Segment the registration drop by device, content type and traffic source to find where friction concentrates.
- Test onboarding changes against the early-churn window with an A/B test (see `../ab-test-experimentation`).

## Files
`analysis.py` (data generation, SQL, analysis, charts), `results/` (CSVs and PNGs).

## Run it
`pip install pandas numpy matplotlib` then `python analysis.py`.
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")
print(funnel.to_string(index=False))
print("early churn by channel:\n", early.round(3).to_string())
print("overall paywall->sub:", round(tot.subscribed / tot.hit_paywall, 4))
