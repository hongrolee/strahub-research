# -*- coding: utf-8 -*-
"""
1차 측정에서 나온 '모멘텀 지속' 신호 검증
  A) 연도별 분해      -> 2020~21 불장에서만 나온 착시인가?
  B) 기간 분할 검증   -> 2017~2021에서 발견한 게 2022~2026에도 유효한가?
  C) 비겹침 7일 표본  -> 겹친 구간 때문에 CI가 좁게 나온 것 아닌가?
  D) 실제 백테스트    -> Buy&Hold를 이기는가? (수수료/MDD/Sharpe 포함)
"""
import sqlite3, math
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

DB, TABLE, KST, COST = "data/ohlcv_data_kst.db", "coin_KRW_BTC", 9 * 3600, 0.0016
rng = np.random.default_rng(42)


def load():
    con = sqlite3.connect("file:" + DB + "?mode=ro", uri=True)
    df = pd.read_sql_query(
        "SELECT timestamp, open_price, high_price, low_price, close_price, volume "
        "FROM " + TABLE + " WHERE interval_type='day' ORDER BY timestamp", con)
    con.close()
    df.columns = ["ts", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["ts"] - KST, unit="s", utc=True).dt.tz_localize(None).dt.normalize()
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    today_kst = (datetime.now(timezone.utc) + timedelta(hours=9)).date()
    if df["date"].iloc[-1].date() >= today_kst:
        df = df.iloc[:-1]
    return df.reset_index(drop=True)


def prop_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (ctr - h, ctr + h)


def boot_ci(x, n_boot=2000, block=5):
    x = np.asarray(x, float)
    n = len(x)
    if n < 10:
        return (float("nan"), float("nan"))
    nb = int(np.ceil(n / block))
    s = rng.integers(0, max(n - block + 1, 1), size=(n_boot, nb))
    idx = (s[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    return tuple(np.percentile(x[np.minimum(idx, n - 1)].mean(axis=1), [2.5, 97.5]))


d = load()
d["r1"] = d["close"].pct_change()
delta = d["close"].diff()
gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
d["rsi"] = 100 - 100 / (1 + gain / loss)
ma20, sd20 = d["close"].rolling(20).mean(), d["close"].rolling(20).std()
d["bb_pctb"] = (d["close"] - (ma20 - 2 * sd20)) / (4 * sd20)
d["up_streak"] = (d["r1"] > 0).astype(int).groupby((d["r1"] <= 0).cumsum()).cumsum()
d["fwd1"] = d["close"].shift(-1) / d["close"] - 1
d["fwd7"] = d["close"].shift(-7) / d["close"] - 1
d["year"] = d["date"].dt.year

SIGS = {
    "RSI>70": d["rsi"] > 70,
    "볼린저상단(%b>1)": d["bb_pctb"] > 1,
    "3일연속상승": d["up_streak"] >= 3,
}

# ================================================================ A. 연도별
print("=" * 88)
print("A. 연도별 분해   —  2020~21 불장에서만 나온 착시인가?")
print("=" * 88)
print("    (각 칸: 신호일 수 / 신호일 다음날 평균수익 / 그 해 전체 평균수익)")
print()
hdr = "%-6s%9s" % ("연도", "전체평균")
for s in SIGS:
    hdr += "%22s" % s
print(hdr)
print("-" * 88)
years = sorted(d["year"].unique())
tally = {s: [0, 0] for s in SIGS}          # [초과한 해, 측정된 해]
for y in years:
    yr = d[d["year"] == y]
    base = yr["fwd1"].dropna()
    line = "%-6d%+8.3f%%" % (y, base.mean() * 100)
    for s, mask in SIGS.items():
        sub = yr.loc[mask & yr["fwd1"].notna(), "fwd1"]
        if len(sub) < 5:
            line += "%22s" % ("-" if len(sub) == 0 else "n=%d" % len(sub))
            continue
        diff = sub.mean() - base.mean()
        tally[s][1] += 1
        if diff > 0:
            tally[s][0] += 1
        line += "%22s" % ("n=%-4d %+6.2f%% (%+.2f)" % (len(sub), sub.mean() * 100, diff * 100))
    print(line)
print("-" * 88)
for s, (w, t) in tally.items():
    print("  %-18s  전체평균을 초과한 해: %d / %d" % (s, w, t))

# ================================================================ B. 기간 분할
print()
print("=" * 88)
print("B. 기간 분할 검증   —  앞 기간에서 보이던 게 뒤 기간에도 유효한가?")
print("=" * 88)
SPLIT = pd.Timestamp("2022-01-01")
print("%-20s%-14s%7s%9s%18s%11s" % ("신호", "기간", "표본", "상승률", "95%CI", "평균수익"))
print("-" * 88)
for s, mask in SIGS.items():
    for label, seg in [("2017~2021", d["date"] < SPLIT), ("2022~2026", d["date"] >= SPLIT)]:
        sub = d.loc[mask & seg & d["fwd1"].notna(), "fwd1"]
        basesub = d.loc[seg & d["fwd1"].notna(), "fwd1"]
        if len(sub) < 10:
            continue
        k, m = int((sub > 0).sum()), len(sub)
        lo, hi = prop_ci(k, m)
        mark = " <-- 기준선초과" if lo > (basesub > 0).mean() else ""
        print("%-20s%-14s%7d%8.1f%%  [%5.1f%%,%5.1f%%]%+10.2f%%%s"
              % (s, label, m, k / m * 100, lo * 100, hi * 100, sub.mean() * 100, mark))
    bs = d.loc[(d["date"] >= SPLIT) & d["fwd1"].notna(), "fwd1"]
    print("%-20s%-14s%7d%8.1f%%%29s%+10.2f%%"
          % ("  (해당기간 기준선)", "2022~2026", len(bs), (bs > 0).mean() * 100, "", bs.mean() * 100))
    print("-" * 88)

# ================================================================ C. 비겹침 7일
print()
print("=" * 88)
print("C. 비겹침 7일 표본   —  겹친 구간 때문에 CI가 좁게 나온 것 아닌가?")
print("=" * 88)
print("%-20s%10s%8s%8s%22s" % ("신호", "표본(독립)", "승률", "평균", "95%CI(순기대값)"))
print("-" * 88)
for s, mask in SIGS.items():
    sel = d.loc[mask & d["fwd7"].notna()]
    keep, last = [], -99
    for i in sel.index:                      # 7일 이상 떨어진 진입만 채택
        if i - last >= 7:
            keep.append(i)
            last = i
    sub = d.loc[keep, "fwd7"]
    net = sub - COST
    lo, hi = boot_ci(net.values, block=2)
    flag = "  <== 양수" if lo > 0 else ""
    print("%-20s%10d%7.1f%%%+8.2f%%   [%+6.2f%%,%+6.2f%%]%s"
          % (s, len(sub), (sub > 0).mean() * 100, net.mean() * 100, lo * 100, hi * 100, flag))
print()
print("  * 1차 측정의 7일 표본(503개)은 대부분 겹친 구간 -> 독립 표본은 아래 수준에 불과")

# ================================================================ D. 백테스트
print()
print("=" * 88)
print("D. 실제 백테스트   —  Buy&Hold를 이기는가?   (왕복 0.16%, 신호 다음날 진입)")
print("=" * 88)


def backtest(pos, ret, label):
    pos = pos.fillna(False).astype(float).values
    ret = ret.fillna(0).values
    trades = int(np.abs(np.diff(np.concatenate([[0], pos]))).sum())
    cost = np.abs(np.diff(np.concatenate([[0], pos]))) * (COST / 2)
    daily = pos * ret - cost
    eq = np.cumprod(1 + daily)
    yrs = len(daily) / 365.25
    cagr = eq[-1] ** (1 / yrs) - 1
    act = daily[pos > 0]
    shp = (daily.mean() / daily.std() * math.sqrt(365)) if daily.std() > 0 else 0
    mdd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min()
    print("%-22s%11.1f%%%9.1f%%%8.2f%9.1f%%%8d%9.1f%%"
          % (label, (eq[-1] - 1) * 100, cagr * 100, shp, mdd * 100, trades,
             (pos > 0).mean() * 100))
    return eq


print("%-22s%11s%9s%8s%9s%8s%9s"
      % ("전략", "총수익", "CAGR", "Sharpe", "MDD", "거래수", "시장노출"))
print("-" * 88)
r = d["r1"]
backtest(pd.Series(True, index=d.index), r, "Buy & Hold")
print("-" * 88)
for s, mask in SIGS.items():
    backtest(mask.shift(1), r, s + " (신호일만)")

# 2022 이후만 따로
print("-" * 88)
print("  [2022~2026 구간만]")
d2 = d[d["date"] >= SPLIT].reset_index(drop=True)
r2 = d2["r1"]
backtest(pd.Series(True, index=d2.index), r2, "Buy & Hold")
for s in SIGS:
    m2 = SIGS[s][d["date"] >= SPLIT].reset_index(drop=True)
    backtest(m2.shift(1), r2, s + " (신호일만)")
print("=" * 88)
