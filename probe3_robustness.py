# -*- coding: utf-8 -*-
"""
모멘텀 신호 최종 검증 — 과적합인지 실제 현상인지
  E) 임계값 민감도  -> RSI 70에서만 되면 과적합, 넓게 되면 실제 현상
  F) 체결 가정      -> 종가 즉시 체결은 비현실적. 다음날 시가 진입이면?
  G) 단순 추세필터 대조 -> 그냥 'MA20>MA60'보다 나은가? (없으면 RSI는 무의미)
"""
import sqlite3, math
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

DB, TABLE, KST, COST = "data/ohlcv_data_kst.db", "coin_KRW_BTC", 9 * 3600, 0.0016
SPLIT = pd.Timestamp("2022-01-01")


def load():
    con = sqlite3.connect("file:" + DB + "?mode=ro", uri=True)
    df = pd.read_sql_query(
        "SELECT timestamp, open_price, high_price, low_price, close_price, volume "
        "FROM " + TABLE + " WHERE interval_type='day' ORDER BY timestamp", con)
    con.close()
    df.columns = ["ts", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["ts"] - KST, unit="s", utc=True).dt.tz_localize(None).dt.normalize()
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    if df["date"].iloc[-1].date() >= (datetime.now(timezone.utc) + timedelta(hours=9)).date():
        df = df.iloc[:-1]
    return df.reset_index(drop=True)


d = load()
d["r1"] = d["close"].pct_change()
delta = d["close"].diff()
g = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
l = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
d["rsi"] = 100 - 100 / (1 + g / l)
ma20, sd20 = d["close"].rolling(20).mean(), d["close"].rolling(20).std()
d["bb_pctb"] = (d["close"] - (ma20 - 2 * sd20)) / (4 * sd20)
d["ma20"], d["ma60"] = ma20, d["close"].rolling(60).mean()
d["up_streak"] = (d["r1"] > 0).astype(int).groupby((d["r1"] <= 0).cumsum()).cumsum()

# 3가지 체결 가정
d["ret_cc"] = d["close"].shift(-1) / d["close"] - 1                 # 종가->다음종가 (낙관)
d["ret_oc"] = d["close"].shift(-1) / d["open"].shift(-1) - 1        # 다음시가->다음종가 (현실)
d["ret_oo"] = d["open"].shift(-2) / d["open"].shift(-1) - 1         # 다음시가->다다음시가 (보수)


def stats(mask, retcol, seg=None):
    m = mask if seg is None else (mask & seg)
    sub = d.loc[m & d[retcol].notna(), retcol]
    if len(sub) < 10:
        return None
    return len(sub), (sub > 0).mean(), sub.mean()


def bt(pos, ret, label, sub=None):
    dd = d if sub is None else d[sub].reset_index(drop=True)
    p = pos if sub is None else pos[sub].reset_index(drop=True)
    p = p.fillna(False).infer_objects(copy=False).astype(float).values
    r = dd[ret].fillna(0).values
    ch = np.abs(np.diff(np.concatenate([[0], p])))
    daily = p * r - ch * (COST / 2)
    eq = np.cumprod(1 + daily)
    yrs = len(daily) / 365.25
    mdd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min()
    shp = daily.mean() / daily.std() * math.sqrt(365) if daily.std() > 0 else 0
    print("%-26s%11.1f%%%9.1f%%%8.2f%9.1f%%%8d%9.1f%%"
          % (label, (eq[-1] - 1) * 100, eq[-1] ** (1 / yrs) * 100 - 100, shp, mdd * 100,
             int(ch.sum()), (p > 0).mean() * 100))


# ================================================================ E. 민감도
print("=" * 90)
print("E. 임계값 민감도   —  특정 값에서만 되면 과적합, 넓게 되면 실제 현상")
print("=" * 90)
base_all = d["ret_cc"].dropna()
base_oos = d.loc[(d["date"] >= SPLIT) & d["ret_cc"].notna(), "ret_cc"]
print("  기준선: 전체 상승률 %.1f%% (평균 %+.3f%%)   |   2022~ 상승률 %.1f%% (평균 %+.3f%%)"
      % ((base_all > 0).mean() * 100, base_all.mean() * 100,
         (base_oos > 0).mean() * 100, base_oos.mean() * 100))
print()
print("%-16s%22s%24s" % ("", "전체 2017~2026", "검증구간 2022~2026"))
print("%-16s%8s%7s%8s%9s%7s%8s" % ("임계값", "표본", "상승률", "평균", "표본", "상승률", "평균"))
print("-" * 90)
print("  [RSI 상단]")
for th in [55, 60, 65, 70, 75, 80]:
    mk = d["rsi"] > th
    a = stats(mk, "ret_cc")
    b = stats(mk, "ret_cc", d["date"] >= SPLIT)
    line = "%-16s%8d%6.1f%%%+8.2f%%" % ("RSI > %d" % th, a[0], a[1] * 100, a[2] * 100)
    line += "%9d%6.1f%%%+8.2f%%" % (b[0], b[1] * 100, b[2] * 100) if b else "%26s" % "표본부족"
    print(line)
print("  [볼린저 %b 상단]")
for th in [0.8, 0.9, 1.0, 1.1]:
    mk = d["bb_pctb"] > th
    a = stats(mk, "ret_cc")
    b = stats(mk, "ret_cc", d["date"] >= SPLIT)
    line = "%-16s%8d%6.1f%%%+8.2f%%" % ("%%b > %.1f" % th, a[0], a[1] * 100, a[2] * 100)
    line += "%9d%6.1f%%%+8.2f%%" % (b[0], b[1] * 100, b[2] * 100) if b else "%26s" % "표본부족"
    print(line)
print("  [연속 상승일]")
for th in [2, 3, 4, 5]:
    mk = d["up_streak"] >= th
    a = stats(mk, "ret_cc")
    b = stats(mk, "ret_cc", d["date"] >= SPLIT)
    line = "%-16s%8d%6.1f%%%+8.2f%%" % ("%d일 연속+" % th, a[0], a[1] * 100, a[2] * 100)
    line += "%9d%6.1f%%%+8.2f%%" % (b[0], b[1] * 100, b[2] * 100) if b else "%26s" % "표본부족"
    print(line)
print()
print("  * 임계값을 바꿔도 상승률이 계속 기준선 위면 -> 실제 현상")
print("  * 한두 값에서만 튀면 -> 우연/과적합")

# ================================================================ F. 체결
print()
print("=" * 90)
print("F. 체결 가정별 결과   —  종가 즉시 체결은 비현실적")
print("=" * 90)
SIGS = {"RSI>70": d["rsi"] > 70, "볼린저상단": d["bb_pctb"] > 1, "3일연속상승": d["up_streak"] >= 3}
EXEC = [("ret_cc", "종가->종가 (낙관)"), ("ret_oc", "시가진입->종가청산 (현실)"),
        ("ret_oo", "시가->시가 (보수)")]
print("%-14s%-26s%8s%8s%10s%12s" % ("신호", "체결가정", "표본", "상승률", "평균수익", "비용차감후"))
print("-" * 90)
for s, mk in SIGS.items():
    for col, lab in EXEC:
        r = stats(mk, col)
        if r:
            print("%-14s%-26s%8d%7.1f%%%+9.2f%%%+11.2f%%"
                  % (s, lab, r[0], r[1] * 100, r[2] * 100, (r[2] - COST) * 100))
    print("-" * 90)

# ================================================================ G. 대조군
print()
print("=" * 90)
print("G. 단순 추세필터 대조   —  그냥 'MA20>MA60'보다 나은가?")
print("=" * 90)
print("%-26s%11s%9s%8s%9s%8s%9s"
      % ("전략 (시가진입->종가청산)", "총수익", "CAGR", "Sharpe", "MDD", "거래수", "시장노출"))
print("-" * 90)
allt = pd.Series(True, index=d.index)
bt(allt, "ret_cc", "Buy & Hold")
bt((d["ma20"] > d["ma60"]), "ret_oc", "[대조] MA20>MA60")
for s, mk in SIGS.items():
    bt(mk, "ret_oc", s)
bt((d["rsi"] > 70) & (d["ma20"] > d["ma60"]), "ret_oc", "RSI>70 AND MA20>MA60")
print("-" * 90)
print("  [2022~2026 구간만]")
seg = d["date"] >= SPLIT
bt(allt, "ret_cc", "Buy & Hold", seg)
bt((d["ma20"] > d["ma60"]), "ret_oc", "[대조] MA20>MA60", seg)
for s, mk in SIGS.items():
    bt(mk, "ret_oc", s, seg)
bt((d["rsi"] > 70) & (d["ma20"] > d["ma60"]), "ret_oc", "RSI>70 AND MA20>MA60", seg)
print("=" * 90)
