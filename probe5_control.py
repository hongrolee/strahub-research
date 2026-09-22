# -*- coding: utf-8 -*-
"""
변동성 대조 검증 — RSI>70은 진짜 신호인가, '변동성 큰 날'의 껍데기인가?
  L) 변동성 층화     : 같은 변동성 구간 안에서도 차이가 남는가?
  M) 같은 달 내 비교 : 상승장/하락장 국면 효과를 완전히 제거하면?
  N) 이중 통제       : 같은 달 + 같은 변동성 구간
  O) 맞대결          : RSI>70  vs  고변동성 단독  vs  추세 단독
"""
import sqlite3, math, re
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

DB, KST, COST = "data/ohlcv_data_kst.db", 9 * 3600, 0.0016
MIN_DAYS, RSI_TH, SPLIT = 800, 70, pd.Timestamp("2022-01-01")
rng = np.random.default_rng(42)
con = sqlite3.connect("file:" + DB + "?mode=ro", uri=True)
TODAY = (datetime.now(timezone.utc) + timedelta(hours=9)).date()


def boot_ci(x, n_boot=3000, block=5):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 10:
        return (np.nan, np.nan)
    nb = int(np.ceil(n / block))
    s = rng.integers(0, max(n - block + 1, 1), size=(n_boot, nb))
    idx = (s[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    return tuple(np.percentile(x[np.minimum(idx, n - 1)].mean(axis=1), [2.5, 97.5]))


def prep(tbl):
    df = pd.read_sql_query(
        "SELECT timestamp, close_price FROM %s WHERE interval_type='day' ORDER BY timestamp" % tbl, con)
    if len(df) < MIN_DAYS:
        return None
    df.columns = ["ts", "close"]
    df["date"] = pd.to_datetime(df["ts"] - KST, unit="s", utc=True).dt.tz_localize(None).dt.normalize()
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    if df["date"].iloc[-1].date() >= TODAY:
        df = df.iloc[:-1]
    df = df[df["close"] > 0].reset_index(drop=True)
    if len(df) < MIN_DAYS:
        return None
    df["r1"] = df["close"].pct_change()
    dl = df["close"].diff()
    g = dl.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    l = (-dl.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + g / l)
    df["sig20"] = df["r1"].rolling(20).std()
    df["volq"] = df["sig20"].rolling(252).rank(pct=True)        # 과거 252일 내 변동성 순위
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["fwd1"] = df["close"].shift(-1) / df["close"] - 1
    df["ym"] = df["date"].dt.to_period("M")
    df["sig"] = df["rsi"] > RSI_TH
    df = df.dropna(subset=["fwd1", "volq", "ma60"]).reset_index(drop=True)
    # 같은 달 평균 대비 초과수익 (국면 효과 제거)
    df["exc"] = df["fwd1"] - df.groupby("ym")["fwd1"].transform("mean")
    df["vbin"] = pd.cut(df["volq"], [0, .2, .4, .6, .8, 1.0],
                        labels=["최저20%", "20~40%", "40~60%", "60~80%", "최고20%"])
    return df


btc = prep("coin_KRW_BTC")

# ================================================================ L. 변동성 층화
print("=" * 92)
print("L. 변동성 층화 [BTC]   —  같은 변동성 구간 안에서도 RSI>70이 더 나은가?")
print("=" * 92)
print("%-12s%20s%22s%14s%18s" % ("변동성 구간", "RSI>70 (신호일)", "그 외 (대조군)", "차이", "95%CI(차이)"))
print("%-12s%9s%11s%9s%13s" % ("", "표본", "평균수익", "표본", "평균수익"))
print("-" * 92)
for b in btc["vbin"].cat.categories:
    sub = btc[btc["vbin"] == b]
    a, c = sub[sub["sig"]]["fwd1"], sub[~sub["sig"]]["fwd1"]
    if len(a) < 10:
        print("%-12s%9d%11s" % (b, len(a), "표본부족"))
        continue
    diff = a.mean() - c.mean()
    lo, hi = boot_ci(np.concatenate([a.values - c.mean(), ]))
    mark = "  *" if lo > 0 else ""
    print("%-12s%9d%+10.2f%%%9d%+12.2f%%%+13.2f%%   [%+6.2f%%,%+6.2f%%]%s"
          % (b, len(a), a.mean() * 100, len(c), c.mean() * 100, diff * 100, lo * 100, hi * 100, mark))
print()
print("  * 표시 = 신호일 평균이 그 구간 대조군 평균보다 유의하게 높음")
print("  ! 신호일이 어느 변동성 구간에 몰려 있는지도 확인:")
dist = btc[btc["sig"]]["vbin"].value_counts(normalize=True).reindex(btc["vbin"].cat.categories)
alld = btc["vbin"].value_counts(normalize=True).reindex(btc["vbin"].cat.categories)
print("     %-10s %s" % ("신호일 분포", "  ".join("%s %.0f%%" % (b, v * 100) for b, v in dist.items())))
print("     %-10s %s" % ("전체일 분포", "  ".join("%s %.0f%%" % (b, v * 100) for b, v in alld.items())))

# ================================================================ M. 같은 달
print()
print("=" * 92)
print("M. 같은 달 내 비교 [BTC]   —  상승장/하락장 국면 효과를 완전히 제거하면?")
print("=" * 92)
for label, seg in [("전체 2017~2026", btc["date"] > pd.Timestamp("1900")),
                   ("검증구간 2022~2026", btc["date"] >= SPLIT)]:
    e = btc.loc[seg & btc["sig"], "exc"]
    lo, hi = boot_ci(e.values)
    v = "유의" if lo > 0 else "무의미"
    print("  %-20s 신호일 %4d개   같은달 평균 대비 %+.3f%%   95%%CI [%+.3f%%, %+.3f%%]  -> %s"
          % (label, len(e), e.mean() * 100, lo * 100, hi * 100, v))
print()
print("  * '같은 달 평균 대비'가 0이면 -> RSI는 달 안에서 좋은 날을 못 고른다")
print("    (= 상승장을 고른 것일 뿐, 타이밍 정보는 없음)")

# ================================================================ N. 이중 통제
print()
print("=" * 92)
print("N. 이중 통제 [BTC]   —  같은 달 + 같은 변동성 구간")
print("=" * 92)
print("%-12s%10s%18s%22s" % ("변동성 구간", "신호일", "같은달대비 초과", "95%CI"))
print("-" * 92)
for b in btc["vbin"].cat.categories:
    e = btc.loc[(btc["vbin"] == b) & btc["sig"], "exc"]
    if len(e) < 15:
        print("%-12s%10d%18s" % (b, len(e), "표본부족"))
        continue
    lo, hi = boot_ci(e.values)
    print("%-12s%10d%+17.3f%%   [%+6.3f%%,%+6.3f%%]%s"
          % (b, len(e), e.mean() * 100, lo * 100, hi * 100, "  *" if lo > 0 else ""))

# ================================================================ 전 코인 풀
print()
print("=" * 92)
print("   전 코인(112개) 동일 검증   —  코인별로 계산 후 집계")
print("=" * 92)
tables = sorted(t for t in
                [r[0] for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'coin_KRW_%'")]
                if re.fullmatch(r"coin_KRW_[A-Z0-9]{2,12}", t))
res = []
for t in tables:
    try:
        df = prep(t)
    except Exception:
        continue
    if df is None or df["sig"].sum() < 30:
        continue
    s = df[df["sig"]]
    row = {"tk": t.replace("coin_KRW_", ""), "n": int(df["sig"].sum()),
           "exc": s["exc"].mean(), "raw": s["fwd1"].mean() - df["fwd1"].mean()}
    s2 = df[df["sig"] & (df["date"] >= SPLIT)]
    row["exc_oos"] = s2["exc"].mean() if len(s2) >= 20 else np.nan
    # 최고변동성 구간 안에서만
    hv = df[df["vbin"] == "최고20%"]
    row["exc_hv"] = hv[hv["sig"]]["exc"].mean() if hv["sig"].sum() >= 15 else np.nan
    res.append(row)
P = pd.DataFrame(res)
print("  검증 코인 %d개" % len(P))
print()
print("%-34s%10s%12s%14s" % ("측정", "양(+) 비율", "중앙값", "평균"))
print("-" * 92)
for col, lab in [("raw", "단순 차이 (통제 없음)"),
                 ("exc", "같은 달 대비 초과 (국면 통제)"),
                 ("exc_oos", "같은 달 대비 초과 · 2022~ 만"),
                 ("exc_hv", "같은 달 대비 · 최고변동성 구간만")]:
    v = P[col].dropna()
    if len(v) == 0:
        continue
    print("%-34s%9.1f%%%+11.3f%%%+13.3f%%   (n=%d)"
          % (lab, (v > 0).mean() * 100, v.median() * 100, v.mean() * 100, len(v)))
print()
print("  * 코인 간 상관 0.45 -> 실질 독립 표본 약 2개. '몇 %% 코인이 양수'는 증거력이 약함")

# ================================================================ O. 맞대결
print()
print("=" * 92)
print("O. 맞대결 [BTC]   —  RSI>70  vs  고변동성 단독  vs  추세 단독")
print("=" * 92)
CAND = {
    "RSI > 70": btc["sig"],
    "변동성 최고20%": btc["volq"] > 0.8,
    "추세 MA20>MA60": btc["ma20"] > btc["ma60"],
    "RSI>70 + 변동성최고20%": btc["sig"] & (btc["volq"] > 0.8),
    "RSI>70 + 변동성 그외": btc["sig"] & (btc["volq"] <= 0.8),
}
print("%-24s%8s%12s%12s%20s" % ("조건", "표본", "평균수익", "같은달대비", "95%CI(같은달대비)"))
print("-" * 92)
for lab, mk in CAND.items():
    sub = btc[mk]
    if len(sub) < 15:
        continue
    lo, hi = boot_ci(sub["exc"].values)
    print("%-24s%8d%+11.3f%%%+11.3f%%   [%+6.3f%%,%+6.3f%%]%s"
          % (lab, len(sub), sub["fwd1"].mean() * 100, sub["exc"].mean() * 100,
             lo * 100, hi * 100, "  *" if lo > 0 else ""))
print("=" * 92)
