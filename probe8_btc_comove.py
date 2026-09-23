# -*- coding: utf-8 -*-
"""
비트코인과 같이 움직이는 코인 — 얼마나, 언제, 그리고 그게 무슨 뜻인가

  A) 표본        : 균형 패널 (기간이 제각각이면 순위를 비교할 수 없다)
  B) 수익률 정의 : 로그 vs 단순, 무거래일 처리를 두 방식으로 대조
  C) 순위        : 상관 · 스피어만 · 블록부트스트랩 신뢰구간과 순위구간 · 베타 · 변동성비
  D) 유동성      : 거래가 적으면 상관이 «낮게» 측정된다. 그 크기를 잰다
  E) 다기간      : 1 / 3 / 7 / 30일. 주기를 늘려 상관이 오르면 체결 시차다
  F) 언제 붙나   : 상승일 vs 하락일 · 극단일 제외 · 연도별 · 동반하락 비율
  G) 분산        : ★ sigma_p(N) 과 PC1 — «분산이 되는가» 의 진짜 측정치
  H) 위약 인자   : BTC 자리에 다른 코인을 넣어도 같은 그림이 나오는가
  I) 원화 통제   : 동조의 일부가 단지 «둘 다 원화 표시» 여서인가
  J) 생존편향    : 상장폐지 코인을 넣으면 달라지는가 (별도 CSV 가 있을 때만)
  K) 대조군      : 스테이블코인 — 심어 놓지 않은 대조군

pandas 와 numpy 만 쓴다. scipy 는 필요 없다 (스피어만은 순위 후 피어슨으로 직접 계산).

    python fetch_data.py --all-coins     # 먼저 (약 12분)
    python probe8_btc_comove.py

숫자를 그대로 재현하려면 구간을 고정해야 한다. WIN_START / WIN_END 가 그 고정값이다.
"""
import hashlib
import json
import os
import re
import sqlite3
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "data", "ohlcv_data_kst.db")
DEAD_CSV = os.path.join(HERE, "data", "delisted_daily.csv")   # 있으면 J 절을 돌린다
OUT = os.path.join(HERE, "outputs")
KST = 9 * 3600

# ── 고정값 (바꾸면 글의 숫자가 달라진다) ──────────────────────────────
WIN_START = "2024-09-01"
WIN_END = "2026-09-23"
COVER = 0.95          # 공통 거래일의 몇 %를 채워야 순위 대상인가
MIN_DAYS_LOOSE = 400  # 순위에는 못 들어가도 참고로 보는 하한
BLOCK = 20            # 블록 부트스트랩 블록 길이 (변동성 뭉침을 보존한다)
BOOT = 500
SEED = 20260924
HORIZONS = [1, 3, 7, 30]
PORT_N = [1, 2, 3, 5, 10, 20, 50, 100]
PORT_DRAWS = 300
STABLE = ["USDT", "USDC", "USDE", "USDG", "USDS", "USD1", "DAI"]

rng = np.random.default_rng(SEED)
line = "=" * 92


def head(t):
    print()
    print(line)
    print(t)
    print(line)


# ════════════════════════════════════════════════════════ 데이터 적재
def load_panel(con):
    """코인별 종가·거래대금을 넓은 표 둘로 만든다 (날짜 x 티커)."""
    tabs = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'coin_KRW_%'")]
    tabs = sorted(t for t in tabs if re.fullmatch(r"coin_KRW_[A-Z0-9]{2,12}", t))
    close, value = {}, {}
    for t in tabs:
        df = pd.read_sql_query(
            "SELECT timestamp, close_price, value_krw FROM %s WHERE interval_type='day'" % t, con)
        if df.empty:
            continue
        df["d"] = pd.to_datetime(df["timestamp"] - KST, unit="s", utc=True) \
                    .dt.tz_localize(None).dt.normalize()
        df = df.drop_duplicates("d").set_index("d").sort_index()
        tk = t.replace("coin_KRW_", "")
        close[tk] = df["close_price"].where(df["close_price"] > 0)
        value[tk] = df["value_krw"]
    return pd.DataFrame(close), pd.DataFrame(value)


def window(df):
    s, e = pd.Timestamp(WIN_START), pd.Timestamp(WIN_END)
    return df[(df.index >= s) & (df.index <= e)]


# ════════════════════════════════════════════════════════ 통계 도구
def corr_all(R, b):
    """R(T x K) 각 열과 b(T) 의 피어슨 상관을 한 번에. 결측 없는 행렬 전용."""
    Rc = R - R.mean(axis=0)
    bc = b - b.mean()
    num = (Rc * bc[:, None]).sum(axis=0)
    den = np.sqrt((Rc ** 2).sum(axis=0) * (bc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def spearman(a, b):
    """순위로 바꾼 뒤 피어슨. pandas 의 method='spearman' 은 scipy 를 부른다."""
    m = a.notna() & b.notna()
    if m.sum() < 30:
        return np.nan
    return float(np.corrcoef(a[m].rank(), b[m].rank())[0, 1])


def block_index(T, rng_):
    """블록 부트스트랩용 행 번호. 모든 코인에 같은 번호를 써야 횡단면이 어긋나지 않는다."""
    nb = int(np.ceil(T / BLOCK))
    starts = rng_.integers(0, max(1, T - BLOCK + 1), size=nb)
    idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])
    return np.clip(idx[:T], 0, T - 1)


def live_markets():
    try:
        req = urllib.request.Request(
            "https://api.upbit.com/v1/market/all?isDetails=false",
            headers={"User-Agent": "strahub-research/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            mk = json.loads(r.read().decode())
        return {m["market"].replace("KRW-", "") for m in mk if m["market"].startswith("KRW-")}
    except Exception as e:
        print("  업비트 목록 조회 실패(%s) — 생존편향 점검은 건너뜁니다" % type(e).__name__)
        return None


def krw_names():
    try:
        req = urllib.request.Request("https://api.upbit.com/v1/market/all",
                                     headers={"User-Agent": "strahub-research/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            mk = json.loads(r.read().decode())
        return {m["market"].replace("KRW-", ""): m.get("korean_name", "")
                for m in mk if m["market"].startswith("KRW-")}
    except Exception:
        return {}


# ════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(DB):
        print("데이터가 없습니다. 먼저 python fetch_data.py --all-coins 를 돌리세요.")
        return 1
    con = sqlite3.connect("file:" + DB + "?mode=ro", uri=True)
    C_all, V_all = load_panel(con)
    con.close()
    blog = {}          # 글에 싣는 숫자만 따로 모은다

    # ══════════════════════════════════════════════════ A. 표본
    head("A. 표본   —  기간이 제각각인 코인을 한 줄에 세우면 순위가 기간을 섞는다")
    C, V = window(C_all), window(V_all)
    if "BTC" not in C.columns:
        print("  KRW-BTC 가 없습니다.")
        return 1
    cal = C.index[C["BTC"].notna()]            # BTC 가 거래된 날 = 공통 달력
    C, V = C.loc[cal], V.reindex(cal)
    T = len(cal)
    cov = C.notna().sum() / T
    keep = sorted(c for c in C.columns if cov[c] >= COVER)
    loose = sorted(c for c in C.columns if COVER > cov[c] and C[c].notna().sum() >= MIN_DAYS_LOOSE)
    print("  구간            : %s ~ %s  (%d 거래일)" % (WIN_START, WIN_END, T))
    print("  구간에 나타난 코인: %d개" % C.shape[1])
    print("  순위 대상 (%.0f%% 이상 채움): %d개" % (COVER * 100, len(keep)))
    print("  참고 등급 (%d일 이상, 커버리지 미달): %d개" % (MIN_DAYS_LOOSE, len(loose)))
    print("  나머지 (상장이 최근이라 제외) : %d개" % (C.shape[1] - len(keep) - len(loose)))
    live = live_markets()
    if live:
        dead = sorted(set(C.columns) - live)
        print("  현재 원화마켓  : %d개 / 이 DB 에만 있는(폐지 추정) 코인: %d개" % (len(live), len(dead)))
        if len(dead) < 5:
            print("  => 폐지 코인이 거의 없습니다. 생존편향을 감안해서 읽어야 합니다 (J 절).")
    blog["window"] = {"start": WIN_START, "end": WIN_END, "days": int(T),
                      "rows": 0,
                      "seen": int(C.shape[1]), "ranked": len(keep), "loose": len(loose)}

    C = C[keep]
    V = V[keep]

    # ══════════════════════════════════════════════════ B. 수익률 정의
    head("B. 수익률 정의   —  로그냐 단순이냐, 거래가 없던 날을 어떻게 볼 것인가")
    Rlog = np.log(C).diff()
    Rsim = C.pct_change(fill_method=None)
    # 거래대금이 0 인 날 = 사실상 체결이 없던 날. 메우면 «0% 움직임» 을 만들어 낸다.
    notrade = (V.fillna(0) <= 0)
    print("  로그수익률을 기본으로 씁니다 — 3일·7일·30일로 더할 수 있어야 E 절이 성립합니다.")
    print("  거래대금 0 인 날  : 전체의 %.2f%%  (가장 많은 코인 %.1f%%)"
          % (notrade.values.mean() * 100, notrade.mean().max() * 100))
    Rlog_drop = Rlog.mask(notrade)             # 무거래일을 결측으로
    b = Rlog["BTC"]
    c1 = Rlog.apply(lambda s: s.corr(b)).drop("BTC")
    c2 = Rlog_drop.apply(lambda s: s.corr(Rlog_drop["BTC"])).drop("BTC")
    c3 = Rsim.apply(lambda s: s.corr(Rsim["BTC"])).drop("BTC")
    print("  상관 중앙값 — 로그(그대로) %.3f / 로그(무거래일 제외) %.3f / 단순 %.3f"
          % (c1.median(), c2.median(), c3.median()))
    print("  => 세 방식이 갈리지 않습니다. 이 아래로는 «로그, 무거래일 제외» 를 씁니다.")
    R = Rlog_drop
    blog["ret_def"] = {"log": round(float(c1.median()), 4),
                       "log_notrade_dropped": round(float(c2.median()), 4),
                       "simple": round(float(c3.median()), 4),
                       "notrade_pct": round(float(notrade.values.mean() * 100), 3)}

    # 계산의 본체가 되는 결측 없는 행렬
    M = R.dropna()
    tick = [c for c in M.columns if c != "BTC"]
    X = M[tick].to_numpy(float)
    bv = M["BTC"].to_numpy(float)
    Tn = len(M)
    print("  결측 없는 공통 행 : %d일 x %d종목" % (Tn, len(tick)))
    blog["window"]["rows"] = int(Tn)

    # ══════════════════════════════════════════════════ C. 순위
    head("C. 순위   —  상관계수 하나만 적으면 옆자리와 구별되는 것처럼 읽힌다")
    r = corr_all(X, bv)
    beta = np.array([np.cov(X[:, i], bv)[0, 1] / np.var(bv, ddof=1) for i in range(len(tick))])
    sig = X.std(axis=0, ddof=1) / bv.std(ddof=1)
    sp = np.array([spearman(M[t], M["BTC"]) for t in tick])

    boot_r = np.empty((BOOT, len(tick)))
    for i in range(BOOT):
        ix = block_index(Tn, rng)
        boot_r[i] = corr_all(X[ix], bv[ix])
    lo, hi = np.nanpercentile(boot_r, [2.5, 97.5], axis=0)
    boot_rank = (-boot_r).argsort(axis=1).argsort(axis=1) + 1     # 큰 값이 1위
    rk_lo, rk_hi = np.nanpercentile(boot_rank, [2.5, 97.5], axis=0)

    names = krw_names()
    D = pd.DataFrame({
        "ticker": tick,
        "name": [names.get(t, "") for t in tick],
        "r": r, "lo": lo, "hi": hi, "spearman": sp, "beta": beta, "sig_ratio": sig,
        "rank_lo": np.round(rk_lo).astype(int), "rank_hi": np.round(rk_hi).astype(int),
    }).sort_values("r", ascending=False).reset_index(drop=True)
    D.insert(0, "rank", D.index + 1)

    print("  %-5s%-8s%8s%18s%10s%8s%8s" % ("순위", "티커", "상관", "95% 구간", "순위구간", "베타", "변동성비"))
    for _, x in D.head(15).iterrows():
        print("  %-5d%-8s%8.3f   [%5.3f, %5.3f]%7d~%-4d%8.2f%8.2f"
              % (x["rank"], x.ticker, x.r, x.lo, x.hi, x.rank_lo, x.rank_hi, x.beta, x.sig_ratio))
    print("  ...")
    for _, x in D.tail(6).iterrows():
        print("  %-5d%-8s%8.3f   [%5.3f, %5.3f]%7d~%-4d%8.2f%8.2f"
              % (x["rank"], x.ticker, x.r, x.lo, x.hi, x.rank_lo, x.rank_hi, x.beta, x.sig_ratio))
    w = (D["rank_hi"] - D["rank_lo"])
    print()
    print("  상관 중앙값 %.3f   사분위 %.3f ~ %.3f   최소 %.3f  최대 %.3f"
          % (D.r.median(), D.r.quantile(.25), D.r.quantile(.75), D.r.min(), D.r.max()))
    print("  95%% 구간 폭 중앙값 %.3f" % (D.hi - D.lo).median())
    print("  순위구간 폭 중앙값 %.0f위  —  1위 종목의 순위구간 %d~%d위"
          % (w.median(), D.rank_lo.iloc[0], D.rank_hi.iloc[0]))
    print("  => 순위는 «대략 어느 무리» 까지만 말할 수 있습니다. 한 계단 차이는 잡음입니다.")
    blog["rank"] = {
        "median": round(float(D.r.median()), 4),
        "q25": round(float(D.r.quantile(.25)), 4), "q75": round(float(D.r.quantile(.75)), 4),
        "min": round(float(D.r.min()), 4), "max": round(float(D.r.max()), 4),
        "ci_width": round(float((D.hi - D.lo).median()), 4),
        "rank_width": int(w.median()),
        "top_rank_hi": int(D.rank_hi.iloc[0]),
        "rows": [{"t": x.ticker, "n": x["name"], "r": round(float(x.r), 3),
                  "lo": round(float(x.lo), 3), "hi": round(float(x.hi), 3),
                  "rl": int(x.rank_lo), "rh": int(x.rank_hi),
                  "sp": round(float(x.spearman), 3), "b": round(float(x.beta), 2),
                  "sr": round(float(x.sig_ratio), 2)}
                 for _, x in D.head(20).iterrows()],
        "hist": np.histogram(D.r.to_numpy(), bins=np.arange(-0.35, 0.86, 0.05))[0].tolist(),
    }

    # ══════════════════════════════════════════════════ D. 유동성
    head("D. 유동성   —  거래가 적으면 종가가 늦게 따라온다. 상관이 «낮게» 찍힌다")
    vol = V[tick].median()
    zero = (M[tick] == 0).mean()
    D["value_krw"] = vol.reindex(D.ticker).to_numpy()
    D["zero_pct"] = zero.reindex(D.ticker).to_numpy() * 100
    D["value_krw"] = D["value_krw"].fillna(0.0)
    q = pd.qcut(D["value_krw"].rank(method="first"), 4,
                labels=["1분위(적음)", "2분위", "3분위", "4분위(많음)"])
    g = D.groupby(q, observed=True)["r"].agg(["count", "median", "min", "max"])
    print("  거래대금 4분위별 상관")
    for k, x in g.iterrows():
        print("    %-12s n=%3d  중앙값 %.3f   (%.3f ~ %.3f)" % (k, x["count"], x["median"], x["min"], x["max"]))
    rr = np.corrcoef(np.log10(D["value_krw"].clip(lower=1)), D["r"])[0, 1]
    print("  log(거래대금) 과 상관의 상관계수 : %.3f" % rr)
    print("  0.0%% 수익률 일 비율 — 중앙값 %.1f%%, 최대 %.1f%%" % (D.zero_pct.median(), D.zero_pct.max()))
    blog["liquidity"] = {
        "quartile": [{"k": str(k), "n": int(x["count"]), "median": round(float(x["median"]), 3)}
                     for k, x in g.iterrows()],
        "r_vs_logvalue": round(float(rr), 3),
        "zero_median": round(float(D.zero_pct.median()), 2),
    }
    # 산점도는 E 절의 30일 상관이 채워진 뒤에 담는다 (아래에서 이어 붙인다)

    # ══════════════════════════════════════════════════ E. 다기간
    head("E. 다기간   —  주기를 늘려 상관이 오르면 «따로 논다» 가 아니라 체결 시차다")
    hz, hzn = {}, {}
    for k in HORIZONS:
        if k == 1:
            Rk = M
        else:
            Rk = M.rolling(k).sum().iloc[k - 1::k]          # 로그라서 더하면 된다
        bk = Rk["BTC"].to_numpy(float)
        Xk = Rk[tick].to_numpy(float)
        ck = pd.Series(corr_all(Xk, bk), index=tick)
        hz[k], hzn[k] = ck, len(Rk)
        print("  %2d일  표본 %4d   중앙값 %.3f   사분위 %.3f ~ %.3f"
              % (k, len(Rk), ck.median(), ck.quantile(.25), ck.quantile(.75)))
    lowq = D.nsmallest(20, "r").ticker.tolist()
    blog["rank"]["rows_low"] = [
        {"t": x.ticker, "n": x["name"], "r": round(float(x.r), 3),
         "lo": round(float(x.lo), 3), "hi": round(float(x.hi), 3),
         "rl": int(x.rank_lo), "rh": int(x.rank_hi),
         "sp": round(float(x.spearman), 3), "b": round(float(x.beta), 2),
         "sr": round(float(x.sig_ratio), 2)}
        for _, x in D.tail(8).iterrows()]
    print("  1일 상관이 가장 낮은 20종목만 보면 : 1일 %.3f -> 7일 %.3f -> 30일 %.3f"
          % (hz[1][lowq].median(), hz[7][lowq].median(), hz[30][lowq].median()))
    print("  => 하위 무리의 «낮은 상관» 이 주기를 늘리면 얼마나 메워지는지가 판단 근거입니다.")
    for k in HORIZONS:
        D["r%dd" % k] = hz[k].reindex(D.ticker).to_numpy()
    blog["horizon"] = {"all": {str(k): round(float(hz[k].median()), 4) for k in HORIZONS},
                       "low20": {str(k): round(float(hz[k][lowq].median()), 4) for k in HORIZONS},
                       "n": {str(k): int(hzn[k]) for k in HORIZONS}}
    # 산점도 — 거래대금 대비 1일 상관과 30일 상관을 같이 담는다.
    # 거래가 적어 상관이 «낮게 찍힌» 것이라면 주기를 늘릴 때 메워져야 한다.
    blog["liquidity"]["scatter"] = [
        {"v": round(float(np.log10(max(x.value_krw, 1))), 2),
         "r": round(float(x.r), 3), "r30": round(float(x.r30d), 3),
         "z": round(float(x.zero_pct), 1), "t": x.ticker}
        for _, x in D.iterrows()]

    # ══════════════════════════════════════════════════ F. 언제 붙나
    head("F. 언제 붙나   —  오를 때보다 내릴 때 더 붙는가")
    q05, q95 = np.percentile(bv, [5, 95])
    calm = (bv > q05) & (bv < q95)
    r_calm = pd.Series(corr_all(X[calm], bv[calm]), index=tick)
    print("  전체            중앙값 %.3f" % D.r.median())
    print("  BTC 극단일(상·하위 5%%) 제외  중앙값 %.3f   (%d일 제외)"
          % (r_calm.median(), int((~calm).sum())))
    up, dn = bv > 0, bv < 0
    r_up = pd.Series(corr_all(X[up], bv[up]), index=tick)
    r_dn = pd.Series(corr_all(X[dn], bv[dn]), index=tick)
    print("  BTC 오른 날 %.3f  /  내린 날 %.3f" % (r_up.median(), r_dn.median()))
    print("  ⚠ 한쪽 부호만 골라 재면 상관이 구조적으로 낮게 나옵니다. 두 값끼리만 비교해야 합니다.")
    worst = np.argsort(bv)[:max(1, int(len(bv) * 0.05))]
    best = np.argsort(bv)[-max(1, int(len(bv) * 0.05)):]
    co_dn = (X[worst] < 0).mean(axis=1)
    co_up = (X[best] > 0).mean(axis=1)
    print("  BTC 최악 5%% 일 : 같이 내린 종목 비율 평균 %.1f%%  (중앙값 %.1f%%)"
          % (co_dn.mean() * 100, np.median(co_dn) * 100))
    print("  BTC 최고 5%% 일 : 같이 오른 종목 비율 평균 %.1f%%  (중앙값 %.1f%%)"
          % (co_up.mean() * 100, np.median(co_up) * 100))
    yr = {}
    for y in sorted({d.year for d in M.index}):
        m = np.array([d.year == y for d in M.index])
        if m.sum() < 120:
            continue
        yr[y] = float(pd.Series(corr_all(X[m], bv[m]), index=tick).median())
        print("  %d년 중앙값 %.3f  (%d일)" % (y, yr[y], m.sum()))
    D["r_calm"] = r_calm.reindex(D.ticker).to_numpy()
    D["r_up"] = r_up.reindex(D.ticker).to_numpy()
    D["r_dn"] = r_dn.reindex(D.ticker).to_numpy()
    blog["when"] = {
        "all": round(float(D.r.median()), 4), "calm": round(float(r_calm.median()), 4),
        "up": round(float(r_up.median()), 4), "down": round(float(r_dn.median()), 4),
        "co_down": round(float(co_dn.mean() * 100), 1), "co_up": round(float(co_up.mean() * 100), 1),
        "by_year": {str(k): round(v, 4) for k, v in yr.items()},
    }

    # ══════════════════════════════════════════════════ G. 분산
    head("G. 분산   —  «상관이 높다» 가 아니라 «종목을 늘리면 변동성이 줄었나» 를 직접 잰다")
    sd1 = float(X.std(axis=0, ddof=1).mean())
    curve = []
    for n in PORT_N:
        if n > len(tick):
            break
        s = []
        for _ in range(PORT_DRAWS):
            pick = rng.choice(len(tick), size=n, replace=False)
            s.append(X[:, pick].mean(axis=1).std(ddof=1))
        curve.append((n, float(np.median(s))))
    print("  등가중 무작위 N종목의 일간 변동성 (표본 %d회씩)" % PORT_DRAWS)
    print("  %4s %12s %14s %10s" % ("N", "실제", "무상관이라면", "차이"))
    for n, v in curve:
        ideal = sd1 / np.sqrt(n)
        print("  %4d %11.3f%% %13.3f%% %9.2f배" % (n, v * 100, ideal * 100, v / ideal))
    floor = curve[-1][1]
    print("  1종목 %.3f%%  ->  %d종목 %.3f%%   (%.0f%%까지만 줄었습니다)"
          % (curve[0][1] * 100, curve[-1][0], floor * 100, floor / curve[0][1] * 100))
    Ccorr = np.corrcoef(X, rowvar=False)
    ev = np.linalg.eigh(Ccorr)[0][::-1]
    pc1 = float(ev[0] / ev.sum())
    print("  PC1 설명력 : %.1f%%  (종목 %d개의 하루 움직임 중 한 방향이 차지하는 몫)"
          % (pc1 * 100, len(tick)))
    blog["diversify"] = {
        "curve": [{"n": n, "real": round(v * 100, 4), "ideal": round(sd1 / np.sqrt(n) * 100, 4)}
                  for n, v in curve],
        "pc1": round(pc1 * 100, 1), "sd1": round(sd1 * 100, 4),
        "floor_pct": round(floor / curve[0][1] * 100, 1),
    }

    # ══════════════════════════════════════════════════ H. 위약 인자
    head("H. 위약 인자   —  BTC 자리에 다른 코인을 넣어도 같은 그림이 나오는가")
    med_btc = float(np.nanmedian(r))
    picks = rng.choice(len(tick), size=min(30, len(tick)), replace=False)
    meds = []
    for i in picks:
        others = np.delete(np.arange(len(tick)), i)
        meds.append(float(np.nanmedian(corr_all(X[:, others], X[:, i]))))
    meds = np.array(meds)
    print("  BTC 를 인자로 썼을 때 중앙 상관        : %.3f" % med_btc)
    print("  다른 코인 30개를 인자로 썼을 때        : 중앙값 %.3f  (%.3f ~ %.3f)"
          % (np.median(meds), meds.min(), meds.max()))
    print("  BTC 보다 높은 인자 : %d / %d개" % (int((meds > med_btc).sum()), len(meds)))
    print("  => «BTC 와 같이 움직인다» 는 사실상 «시장과 같이 움직인다» 입니다.")
    blog["placebo"] = {"btc": round(med_btc, 4), "median": round(float(np.median(meds)), 4),
                       "min": round(float(meds.min()), 4), "max": round(float(meds.max()), 4),
                       "n_above": int((meds > med_btc).sum()), "n": len(meds)}

    # ══════════════════════════════════════════════════ I. 원화 통제
    head("I. 원화 통제   —  동조의 일부가 단지 «둘 다 원화로 표시돼 있어서» 인가")
    if "USDT" in C.columns:
        # M 은 이미 결측을 털어 낸 행렬이다. 여기서 다시 dropna 를 걸면
        # 남는 행이 거의 없어진다.
        u = np.log(C["USDT"]).diff().reindex(M.index)
        Ru = M.sub(u, axis=0).dropna(how="any")
        tick_u = [t for t in tick if t in Ru.columns and t != "USDT"]
        Xu = Ru[tick_u].to_numpy(float)
        bu = Ru["BTC"].to_numpy(float)
        ru = pd.Series(corr_all(Xu, bu), index=tick_u)
        print("  원화 기준        중앙값 %.3f" % D.set_index("ticker").r.reindex(tick_u).median())
        print("  USDT 기준으로 환산 후 중앙값 %.3f   (%d종목, %d일)"
              % (ru.median(), len(tick_u), len(Ru)))
        print("  => 차이가 작으면 «원화라서» 가 아니라 코인끼리 붙어 있는 것입니다.")
        blog["krw_control"] = {"krw": round(float(D.set_index("ticker").r.reindex(tick_u).median()), 4),
                               "usdt": round(float(ru.median()), 4), "n": len(tick_u)}
    else:
        print("  KRW-USDT 가 표본에 없어 건너뜁니다.")

    # ══════════════════════════════════════════════════ J. 생존편향
    head("J. 생존편향   —  «혼자 0으로 간» 코인들이 표본에서 빠져 있다")
    if not os.path.exists(DEAD_CSV):
        print("  data/delisted_daily.csv 가 없습니다 — 이 절은 건너뜁니다.")
        print("  (공개 API 로는 폐지된 코인의 과거 일봉을 받을 수 없어 저장소에 함께 넣어 둡니다.)")
    else:
        dd = pd.read_csv(DEAD_CSV, parse_dates=["date"])
        with open(DEAD_CSV, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()[:16]
        wide = dd.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
        wide = wide.reindex(cal)
        # 폐지 코인은 대부분 구간 도중에 사라졌다. 끝까지 산 종목과 그대로 비교하면
        # 기간이 서로 달라 비교가 성립하지 않는다. 그래서 «둘 다 살아 있던 구간» 으로
        # 잘라 같은 날짜에서 맞비교한다.
        last = wide.apply(lambda s: s.last_valid_index())
        cut = last.dropna().median()
        sub = cal[(cal >= pd.Timestamp(WIN_START)) & (cal <= cut)]
        cov_d = wide.loc[sub].notna().sum() / len(sub)
        dk = [c for c in wide.columns if cov_d[c] >= COVER]
        print("  맞비교 구간 : %s ~ %s  (%d 거래일)"
              % (WIN_START, cut.date(), len(sub)))
        print("  폐지 코인 %d종 중 이 구간을 채우는 것 : %d종" % (wide.shape[1], len(dk)))
        if dk:
            Rd = np.log(wide[dk]).diff().loc[sub]
            Rl = R[tick].loc[sub]
            bsub = R["BTC"].loc[sub]
            ok = Rd.notna().all(axis=1) & Rl.notna().all(axis=1) & bsub.notna()
            bs = bsub[ok].to_numpy(float)
            rd = pd.Series(corr_all(Rd[ok].to_numpy(float), bs), index=dk)
            rl = pd.Series(corr_all(Rl[ok].to_numpy(float), bs), index=tick)
            print("  폐지된 코인 %3d종  상관 중앙값 %.3f   (%.3f ~ %.3f)"
                  % (len(rd), rd.median(), rd.min(), rd.max()))
            print("  살아남은 코인 %3d종  상관 중앙값 %.3f" % (len(rl), rl.median()))
            print("  같은 구간 수익률 중앙값 — 폐지 %+.3f%%/일   생존 %+.3f%%/일"
                  % (Rd[ok].mean().median() * 100, Rl[ok].mean().median() * 100))
            print("  => 폐지 코인을 빼고 보면 상관은 %s 보이고, 개별 종목이 혼자 무너지는"
                  % ("더 낮게" if rd.median() > rl.median() else "더 높게"))
            print("     위험은 표본에서 사라집니다. 이 글의 표는 «살아남은 것만» 본 표입니다.")
            print("  CSV sha256(앞 16자리) : %s" % sha)
            blog["survivor"] = {
                "cut": str(cut.date()), "days": int(ok.sum()),
                "dead_n": int(len(rd)), "dead_median": round(float(rd.median()), 4),
                "live_n": int(len(rl)), "live_median": round(float(rl.median()), 4),
                "dead_ret": round(float(Rd[ok].mean().median() * 100), 4),
                "live_ret": round(float(Rl[ok].mean().median() * 100), 4),
                "sha": sha}

    # ══════════════════════════════════════════════════ K. 대조군
    head("K. 대조군   —  스테이블코인은 심어 놓은 것이 아니라 원래 시장에 있던 대조군")
    S = D[D.ticker.isin(STABLE)]
    if len(S):
        print("  %-8s%8s%18s%10s" % ("티커", "상관", "95% 구간", "0% 비율"))
        for _, x in S.iterrows():
            print("  %-8s%8.3f   [%5.3f, %5.3f]%9.1f%%" % (x.ticker, x.r, x.lo, x.hi, x.zero_pct))
        print("  => 값이 0 근처가 아니라 음수인 것이 정상입니다. 원화로 코인을 살 때")
        print("     스테이블코인은 반대편에 섭니다. 이 눈금이 «측정이 되고 있다» 는 증거입니다.")
        blog["stable"] = [{"t": x.ticker, "n": x["name"], "r": round(float(x.r), 3),
                           "lo": round(float(x.lo), 3), "hi": round(float(x.hi), 3)}
                          for _, x in S.iterrows()]

    # ══════════════════════════════════════════════════ 이동 상관
    head("L. 90일 이동 상관   —  한 숫자로 요약되지 않는다")
    roll = []
    W = 90
    for i in range(W, Tn + 1, 5):
        sl = slice(i - W, i)
        rc = corr_all(X[sl], bv[sl])
        roll.append((M.index[i - 1], float(np.nanmedian(rc)),
                     float(np.nanpercentile(rc, 10)), float(np.nanpercentile(rc, 90))))
    rs = pd.DataFrame(roll, columns=["date", "med", "p10", "p90"])
    print("  중앙값의 범위 : %.3f ~ %.3f" % (rs.med.min(), rs.med.max()))
    print("  가장 낮았던 때 %s   가장 높았던 때 %s"
          % (rs.loc[rs.med.idxmin(), "date"].date(), rs.loc[rs.med.idxmax(), "date"].date()))
    blog["rolling"] = {
        "min": round(float(rs.med.min()), 3), "max": round(float(rs.med.max()), 3),
        "series": [{"d": d.strftime("%Y-%m-%d"), "m": round(m, 3),
                    "l": round(l, 3), "h": round(h, 3)}
                   for d, m, l, h in roll],
    }

    # ══════════════════════════════════════════════════ 저장
    D.to_csv(os.path.join(OUT, "btc_comove.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(OUT, "btc_comove_blog.json"), "w", encoding="utf-8") as f:
        json.dump(blog, f, ensure_ascii=False, indent=1)
    print()
    print(line)
    print("  저장: outputs/btc_comove.csv        (종목별 전체 수치)")
    print("  저장: outputs/btc_comove_blog.json  (글에 싣는 숫자만)")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
