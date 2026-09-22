# -*- coding: utf-8 -*-
"""
검정 방법 자체를 검증 — 가짜 가격에 같은 파이프라인을 돌리면?

probe6에서 조건 191개 중 40개가 '통제 후에도 살아남았다'(우연 기댓값 5.4개).
그런데 살아남은 목록이 서로 모순된다 (MA 상위=+0.34%, MACD 상위=-0.09%, 둘 다 상승추세).
-> 신호가 진짜인지, 검정 방법이 부푸는 것인지 구분해야 한다.

방법: 예측 가능성이 '설계상 0'인 가짜 가격을 만들어 똑같이 돌린다.
      일별 OHLC 비율을 20일 블록 단위로 재배열 -> 변동성 뭉침과 캔들 모양은 보존,
      장기적 예측 가능성만 파괴. 여기서도 20~40개가 살아남으면 방법의 문제다.
"""
import sqlite3, math, warnings, sys
import numpy as np
import pandas as pd
import talib
from talib import abstract
from datetime import datetime, timezone, timedelta

warnings.filterwarnings("ignore")
DB, KST = "data/ohlcv_data_kst.db", 9 * 3600
MIN_N, TCRIT, BLOCK, K = 40, 1.96, 20, 20
SKIP_GROUPS = {"Math Operators", "Math Transform", "Price Transform"}
SKIP_FUNCS = {"MAVP"}
rng = np.random.default_rng(7)

# ---------------------------------------------------------------- 원본
con = sqlite3.connect("file:" + DB + "?mode=ro", uri=True)
d0 = pd.read_sql_query(
    "SELECT timestamp, open_price, high_price, low_price, close_price, volume "
    "FROM coin_KRW_BTC WHERE interval_type='day' ORDER BY timestamp", con)
con.close()
d0.columns = ["ts", "open", "high", "low", "close", "volume"]
d0["date"] = pd.to_datetime(d0["ts"] - KST, unit="s", utc=True).dt.tz_localize(None).dt.normalize()
d0 = d0.drop_duplicates("date").sort_values("date").reset_index(drop=True)
if d0["date"].iloc[-1].date() >= (datetime.now(timezone.utc) + timedelta(hours=9)).date():
    d0 = d0.iloc[:-1]
if d0["volume"].iloc[-1] < 0.2 * d0["volume"].iloc[-31:-1].median():
    d0 = d0.iloc[:-1]
d0 = d0[d0["close"] > 0].reset_index(drop=True)
DATES = d0["date"].to_numpy()
N0 = len(d0)

# 전일 종가 대비 비율로 분해 (가격 수준을 제거)
pc = d0["close"].shift(1)
RAT = np.column_stack([
    (d0["open"] / pc).to_numpy(), (d0["high"] / pc).to_numpy(),
    (d0["low"] / pc).to_numpy(), (d0["close"] / pc).to_numpy(),
    d0["volume"].to_numpy()])[1:]              # 첫 행은 NaN
NR = len(RAT)


def rebuild(rat, start):
    """비율 시퀀스로 OHLCV 재구성"""
    c = start * np.cumprod(rat[:, 3])
    prev = np.concatenate([[start], c[:-1]])
    return {"open": prev * rat[:, 0], "high": prev * rat[:, 1],
            "low": prev * rat[:, 2], "close": c, "volume": rat[:, 4]}


def surrogate():
    """20일 블록 재배열 — 변동성 뭉침·캔들 모양 보존, 예측 가능성만 파괴"""
    nb = int(np.ceil(NR / BLOCK))
    st = rng.integers(0, NR - BLOCK, nb)
    r = np.concatenate([RAT[s:s + BLOCK] for s in st])[:NR]
    return rebuild(r, float(d0["close"].iloc[0]))


# ---------------------------------------------------------------- 파이프라인
GROUPS = {g: f for g, f in talib.get_function_groups().items() if g not in SKIP_GROUPS}


def build_conditions(inp):
    n = len(inp["close"])
    series, patterns = {}, {}
    for grp, funcs in GROUPS.items():
        for fn in funcs:
            if fn in SKIP_FUNCS:
                continue
            try:
                out = abstract.Function(fn)(inp)
            except Exception:
                continue
            outs = out if isinstance(out, list) else [out]
            nms = getattr(abstract.Function(fn), "output_names", None) or ["out"]
            for i, arr in enumerate(outs):
                arr = np.asarray(arr, dtype=float)
                if arr.shape != (n,) or np.all(~np.isfinite(arr)):
                    continue
                nm = fn if len(outs) == 1 else "%s.%s" % (fn, nms[i] if i < len(nms) else i)
                if grp == "Pattern Recognition":
                    patterns[nm] = arr
                else:
                    fin = np.isfinite(arr) & (inp["close"] > 0)
                    if fin.sum() > 100:
                        ratio = np.nanmedian(np.abs(arr[fin]) / inp["close"][fin])
                        if 0.3 < ratio < 3.0:
                            arr = arr / inp["close"] - 1
                    series[nm] = arr
    conds = {}
    for nm, arr in series.items():
        r = pd.Series(arr).rolling(252, min_periods=120).rank(pct=True).to_numpy()
        for m, lab in [(r > 0.8, "상위20%"), (r < 0.2, "하위20%")]:
            m = np.nan_to_num(m, nan=False).astype(bool)
            if m.sum() >= MIN_N:
                conds["%s %s" % (nm, lab)] = m
    for nm, arr in patterns.items():
        for sgn, lab in [(1, "상승형"), (-1, "하락형")]:
            m = (arr * sgn) > 0
            if m.sum() >= MIN_N:
                conds["%s %s" % (nm.replace("CDL", ""), lab)] = m
    return conds


def batch_t(x, g=10):
    n = len(x)
    k = n // g
    if k < 8:
        s = x.std(ddof=1)
        return 0.0 if s == 0 else x.mean() / (s / math.sqrt(n))
    b = x[:k * g].reshape(k, g).mean(axis=1)
    s = b.std(ddof=1)
    return 0.0 if s == 0 else b.mean() / (s / math.sqrt(k))


def evaluate(inp, dates):
    """조건 생성 -> 국면 통제 -> 생존 개수"""
    close = inp["close"]
    fwd = np.concatenate([close[1:] / close[:-1] - 1, [np.nan]])
    df = pd.DataFrame({"fwd": fwd, "ym": pd.Series(dates).dt.to_period("M")})
    ok = np.isfinite(df["fwd"].to_numpy())
    df.loc[~ok, "fwd"] = np.nan
    exc = (df["fwd"] - df.groupby("ym")["fwd"].transform("mean")).to_numpy()
    raw = df["fwd"].to_numpy() - np.nanmean(df["fwd"].to_numpy())
    conds = build_conditions(inp)
    sr = se = 0
    tot = 0
    survivors = []
    for nm, m in conds.items():
        idx = np.flatnonzero(m & np.isfinite(exc))
        if len(idx) < MIN_N:
            continue
        tot += 1
        if abs(batch_t(raw[idx])) > TCRIT:
            sr += 1
        te = batch_t(exc[idx])
        if abs(te) > TCRIT:
            se += 1
            survivors.append((nm, m, te, exc[idx].mean()))
    return tot, sr, se, survivors


def n_groups(survivors, thr=0.7):
    """신호일이 70% 이상 겹치면 같은 것으로 묶어 종류 수를 센다"""
    ms = [s[1] for s in survivors]
    p = list(range(len(ms)))

    def find(a):
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            u = np.logical_or(ms[i], ms[j]).sum()
            if u and np.logical_and(ms[i], ms[j]).sum() / u >= thr:
                ra, rb = find(i), find(j)
                if ra != rb:
                    p[ra] = rb
    return len({find(i) for i in range(len(ms))})


# ---------------------------------------------------------------- 실행
real_inp = {k: d0[k].to_numpy(float) for k in ["open", "high", "low", "close", "volume"]}
dts = pd.Series(DATES)
tot, sr, se, surv = evaluate(real_inp, dts)
g_real = n_groups(surv)
print("=" * 80)
print("진짜 비트코인 가격")
print("=" * 80)
print("  조건 %d개 중  통제 전 생존 %d개 / 통제 후 생존 %d개 (%d종류)"
      % (tot, sr, se, g_real))

print()
print("=" * 80)
print("가짜 가격 %d개 — 예측 가능성이 설계상 0인 데이터" % K)
print("=" * 80)
print("%6s%10s%12s%14s%12s" % ("회차", "조건수", "통제전생존", "통제후생존", "종류수"))
print("-" * 80)
fr, fe, fg = [], [], []
sdts = pd.Series(DATES[1:])
for k in range(K):
    inp = surrogate()
    t2, r2, e2, s2 = evaluate(inp, sdts)
    g2 = n_groups(s2)
    fr.append(r2); fe.append(e2); fg.append(g2)
    print("%6d%10d%12d%14d%12d" % (k + 1, t2, r2, e2, g2))
    sys.stdout.flush()

fr, fe, fg = np.array(fr), np.array(fe), np.array(fg)
print("-" * 80)
print("%6s%10s%12.1f%14.1f%12.1f" % ("평균", "", fr.mean(), fe.mean(), fg.mean()))
print("%6s%10s%12.1f%14.1f%12.1f" % ("표준편차", "", fr.std(), fe.std(), fg.std()))

print()
print("=" * 80)
print("판정")
print("=" * 80)
z_e = (se - fe.mean()) / fe.std() if fe.std() > 0 else 0
z_g = (g_real - fg.mean()) / fg.std() if fg.std() > 0 else 0
print("  통제 후 생존 개수 : 진짜 %d개  vs  가짜 %.1f ± %.1f개   ->  z = %+.2f"
      % (se, fe.mean(), fe.std(), z_e))
print("  중복 제거 종류 수 : 진짜 %d개  vs  가짜 %.1f ± %.1f개   ->  z = %+.2f"
      % (g_real, fg.mean(), fg.std(), z_g))
print("  가짜가 진짜 이상으로 나온 횟수 : %d / %d" % (int((fe >= se).sum()), K))
print()
if abs(z_g) < 2:
    print("  => 가짜 가격에서도 같은 수가 살아남는다. 살아남은 지표들은 신호가 아니라")
    print("     검정 방법이 만들어낸 것이다. 기술적 분석 지표에 내일 방향 정보가 없다.")
else:
    print("  => 진짜가 가짜보다 확실히 많이 살아남았다. 개별 검증으로 넘어갈 근거가 있다.")
print("=" * 80)
