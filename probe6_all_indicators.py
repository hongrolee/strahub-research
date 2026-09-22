# -*- coding: utf-8 -*-
"""
전 지표 일괄 검증 — 기술적 분석 전체를 같은 필터에 통과시키면 몇 개가 살아남나?

  TA-Lib 지표 + 캔들패턴 61종을 전부 조건으로 변환한 뒤
  probe5와 동일한 국면 통제(같은 달 안에서 비교)를 적용한다.

  핵심: 100개를 시험하면 아무 정보가 없어도 5개쯤은 우연히 살아남는다.
        그래서 '살아남은 개수'를 '우연히 기대되는 개수'와 비교해야 한다.
        귀무분포는 신호 배열을 원형 회전(clustering은 보존, 정렬만 파괴)해서 만든다.
"""
import sqlite3, math, warnings, os
import numpy as np
import pandas as pd
import talib
from talib import abstract
from datetime import datetime, timezone, timedelta

warnings.filterwarnings("ignore")
DB, KST = "data/ohlcv_data_kst.db", 9 * 3600
MIN_N, ROT, TCRIT = 40, 200, 1.96
rng = np.random.default_rng(42)

SKIP_GROUPS = {"Math Operators", "Math Transform", "Price Transform"}
SKIP_FUNCS = {"MAVP"}                      # 가변 주기 — 별도 입력 필요

# ---------------------------------------------------------------- 데이터
con = sqlite3.connect("file:" + DB + "?mode=ro", uri=True)
d = pd.read_sql_query(
    "SELECT timestamp, open_price, high_price, low_price, close_price, volume "
    "FROM coin_KRW_BTC WHERE interval_type='day' ORDER BY timestamp", con)
con.close()
d.columns = ["ts", "open", "high", "low", "close", "volume"]
d["date"] = pd.to_datetime(d["ts"] - KST, unit="s", utc=True).dt.tz_localize(None).dt.normalize()
d = d.drop_duplicates("date").sort_values("date").reset_index(drop=True)
if d["date"].iloc[-1].date() >= (datetime.now(timezone.utc) + timedelta(hours=9)).date():
    d = d.iloc[:-1]
med = d["volume"].iloc[-31:-1].median()
if d["volume"].iloc[-1] < 0.2 * med:
    d = d.iloc[:-1]
d = d[d["close"] > 0].reset_index(drop=True)

d["fwd1"] = d["close"].shift(-1) / d["close"] - 1
d["ym"] = d["date"].dt.to_period("M")
d = d.dropna(subset=["fwd1"]).reset_index(drop=True)
d["exc"] = d["fwd1"] - d.groupby("ym")["fwd1"].transform("mean")   # 같은 달 평균 대비

inputs = {k: d[k].to_numpy(dtype=float) for k in ["open", "high", "low", "close", "volume"]}
FWD = d["fwd1"].to_numpy(float)
EXC = d["exc"].to_numpy(float)
RAW = FWD - FWD.mean()                                             # 통제 전 기준
N = len(d)
print("데이터: %s ~ %s  (%d일)" % (d["date"].iloc[0].date(), d["date"].iloc[-1].date(), N))

# ---------------------------------------------------------------- 지표 생성
series = {}      # 이름 -> 연속형 지표 배열
patterns = {}    # 이름 -> 캔들패턴 배열(-100/0/100)
groups = talib.get_function_groups()
n_fail = 0
for grp, funcs in groups.items():
    if grp in SKIP_GROUPS:
        continue
    for fn in funcs:
        if fn in SKIP_FUNCS:
            continue
        try:
            out = abstract.Function(fn)(inputs)
        except Exception:
            n_fail += 1
            continue
        outs = out if isinstance(out, list) else [out]
        names = getattr(abstract.Function(fn), "output_names", None) or ["out"]
        for i, arr in enumerate(outs):
            arr = np.asarray(arr, dtype=float)
            if arr.shape != (N,) or np.all(~np.isfinite(arr)):
                continue
            nm = fn if len(outs) == 1 else "%s.%s" % (fn, names[i] if i < len(names) else i)
            if grp == "Pattern Recognition":
                patterns[nm] = arr
            else:
                # 가격 수준 지표는 close 대비 비율로 정상화 (BTC는 9년간 100배 변동)
                fin = np.isfinite(arr) & (inputs["close"] > 0)
                if fin.sum() > 100:
                    ratio = np.nanmedian(np.abs(arr[fin]) / inputs["close"][fin])
                    if 0.3 < ratio < 3.0:
                        arr = arr / inputs["close"] - 1
                series[nm] = arr

print("지표 %d개 / 캔들패턴 %d개 생성 (실패 %d)" % (len(series), len(patterns), n_fail))

# ---------------------------------------------------------------- 조건 생성
def trailing_rank(a, win=252):
    """과거 win일 내 백분위 — 미래 정보 없음"""
    return pd.Series(a).rolling(win, min_periods=120).rank(pct=True).to_numpy()

conds = {}
for nm, arr in series.items():
    r = trailing_rank(arr)
    hi, lo = (r > 0.8), (r < 0.2)
    if np.nansum(hi) >= MIN_N:
        conds["%s 상위20%%" % nm] = np.nan_to_num(hi, nan=False).astype(bool)
    if np.nansum(lo) >= MIN_N:
        conds["%s 하위20%%" % nm] = np.nan_to_num(lo, nan=False).astype(bool)
for nm, arr in patterns.items():
    for sgn, lab in [(1, "상승형"), (-1, "하락형")]:
        m = (arr * sgn) > 0
        if m.sum() >= MIN_N:
            conds["%s %s" % (nm.replace("CDL", ""), lab)] = m
print("검정 대상 조건: %d개\n" % len(conds))

# ---------------------------------------------------------------- 검정
def batch_t(x, g=10):
    """블록 평균 t통계 — 연속된 날의 상관을 보정"""
    n = len(x)
    k = n // g
    if k < 8:
        s = x.std(ddof=1)
        return 0.0 if s == 0 else x.mean() / (s / math.sqrt(n))
    b = x[:k * g].reshape(k, g).mean(axis=1)
    s = b.std(ddof=1)
    return 0.0 if s == 0 else b.mean() / (s / math.sqrt(k))

rows = []
for nm, m in conds.items():
    idx = np.flatnonzero(m)
    if len(idx) < MIN_N:
        continue
    rows.append({"cond": nm, "n": len(idx),
                 "raw": RAW[idx].mean(), "t_raw": batch_t(RAW[idx]),
                 "exc": EXC[idx].mean(), "t_exc": batch_t(EXC[idx])})
R = pd.DataFrame(rows)
M = len(R)

# 귀무분포: 원형 회전 (뭉친 구조는 보존, 수익률과의 정렬만 파괴)
#   중요 — 한 번의 추첨에서 모든 조건에 '같은' 회전량을 적용한다.
#   조건마다 다른 난수를 주면 지표들끼리의 중복(SMA/EMA/WMA가 사실상 같은 신호)이
#   깨져서, 우연 기댓값의 분산이 실제보다 훨씬 작게 나온다.
MASKS = [m for m in conds.values() if m.sum() >= MIN_N]
null_raw, null_exc = [], []
for _ in range(ROT):
    off = int(rng.integers(200, N - 200))
    cr = ce = 0
    for m in MASKS:
        idx = np.flatnonzero(np.roll(m, off))
        if abs(batch_t(RAW[idx])) > TCRIT:
            cr += 1
        if abs(batch_t(EXC[idx])) > TCRIT:
            ce += 1
    null_raw.append(cr)
    null_exc.append(ce)
null_raw, null_exc = np.array(null_raw), np.array(null_exc)

sur_raw = int((R["t_raw"].abs() > TCRIT).sum())
sur_exc = int((R["t_exc"].abs() > TCRIT).sum())
pos_exc = int((R["t_exc"] > TCRIT).sum())

print("=" * 84)
print("결과 — 조건 %d개를 전부 같은 기준으로 검정" % M)
print("=" * 84)
print("%-34s%12s%14s%16s" % ("", "살아남은 수", "비율", "우연 기댓값"))
print("-" * 84)
print("%-34s%12d%13.1f%%%10.1f ± %.1f"
      % ("통제 전 (시기를 섞어서)", sur_raw, sur_raw / M * 100, null_raw.mean(), null_raw.std()))
print("%-34s%12d%13.1f%%%10.1f ± %.1f"
      % ("통제 후 (같은 달 안에서)", sur_exc, sur_exc / M * 100, null_exc.mean(), null_exc.std()))
print()
z = (sur_exc - null_exc.mean()) / null_exc.std() if null_exc.std() > 0 else 0
print("  통제 후 생존 %d개 vs 우연 기댓값 %.1f개  ->  z = %+.2f" % (sur_exc, null_exc.mean(), z))
print("  (그중 수익이 플러스 방향인 것: %d개)" % pos_exc)
print()
if abs(z) < 2:
    print("  판정: 우연히 나올 개수와 구분되지 않음")
    print("        -> 기술적 분석 지표 전반에 내일 방향에 대한 정보가 없다")
else:
    print("  판정: (이 스크립트 기준) 우연보다 많이 살아남음")
    print()
    print("  !! 이 판정을 그대로 믿지 마세요. 여기서 쓴 '우연 기댓값'은 틀린 값입니다.")
    print("     지표끼리 겹치는 것(SMA/EMA/WMA/TRIMA 는 사실상 같은 신호)을 독립된 시도로")
    print("     세기 때문에 기댓값이 과소평가됩니다.")
    print("     probe7_surrogate.py 를 돌리면 실제 기댓값은 약 43개로, 진짜 데이터(40개)보다")
    print("     오히려 많습니다. 반드시 probe7 까지 돌린 뒤에 판단하세요.")

print()
print("=" * 84)
print("통제 전 → 통제 후, 상위 12개는 어떻게 되었나")
print("=" * 84)
print("%-40s%7s%11s%11s%9s" % ("조건", "표본", "통제전", "통제후", "t(통제후)"))
print("-" * 84)
for _, r in R.reindex(R["t_raw"].abs().sort_values(ascending=False).index).head(12).iterrows():
    print("%-40s%7d%+10.3f%%%+10.3f%%%9.2f"
          % (r["cond"][:38], r["n"], r["raw"] * 100, r["exc"] * 100, r["t_exc"]))

print()
print("=" * 84)
print("통제 후에도 살아남은 조건 (t > 1.96)")
print("=" * 84)
S = R[R["t_exc"].abs() > TCRIT].reindex(
    R[R["t_exc"].abs() > TCRIT]["t_exc"].abs().sort_values(ascending=False).index)
if len(S) == 0:
    print("  없음")
else:
    print("%-40s%7s%11s%11s%9s" % ("조건", "표본", "통제전", "통제후", "t"))
    print("-" * 84)
    for _, r in S.iterrows():
        print("%-40s%7d%+10.3f%%%+10.3f%%%9.2f"
              % (r["cond"][:38], r["n"], r["raw"] * 100, r["exc"] * 100, r["t_exc"]))
    # ---- 살아남은 것들이 실제로 몇 개의 '서로 다른' 신호인가 ----
    print()
    print("=" * 84)
    print("살아남은 %d개는 실제로 몇 종류인가   (신호일이 70%% 이상 겹치면 같은 것으로 묶음)" % len(S))
    print("=" * 84)
    names = list(S["cond"])
    masks = [conds[n] for n in names]
    parent = list(range(len(names)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = masks[i], masks[j]
            inter = np.logical_and(a, b).sum()
            union = np.logical_or(a, b).sum()
            if union and inter / union >= 0.7:          # 자카드 유사도
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    clusters = {}
    for i, n in enumerate(names):
        clusters.setdefault(find(i), []).append(i)
    order = sorted(clusters.values(), key=lambda g: -max(abs(S.iloc[i]["t_exc"]) for i in g))
    for gi, g in enumerate(order, 1):
        best = max(g, key=lambda i: abs(S.iloc[i]["t_exc"]))
        r = S.iloc[best]
        others = [names[i] for i in g if i != best]
        print("  그룹 %-2d  %-32s n=%-5d t=%+.2f   (같은 신호 %d개)"
              % (gi, r["cond"][:32], r["n"], r["t_exc"], len(g)))
        if others:
            print("          └ %s" % ", ".join(o[:26] for o in others[:7])
                  + (" ..." if len(others) > 7 else ""))
    print()
    print("  이름은 %d개지만 실제로는 %d종류.  우연 기댓값 %.1f ± %.1f개"
          % (len(S), len(order), null_exc.mean(), null_exc.std()))

os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"), exist_ok=True)
R.sort_values("t_exc", ascending=False).to_csv(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "all_indicators.csv"), index=False, encoding="utf-8-sig")
print("\n상세: outputs/all_indicators.csv")
