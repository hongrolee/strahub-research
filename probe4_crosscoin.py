# -*- coding: utf-8 -*-
"""
코인 간 교차검증 — 모멘텀 신호가 BTC에만 있는 건가?
  H) 생존편향 점검   : 상장폐지 코인이 DB에 남아있는가?
  I) 전 코인 적용    : 각 코인별 RSI>70 효과
  J) 독립성 점검     : 알트는 BTC와 같이 움직임 -> 실질 독립 표본 수는?
  K) 검증구간(2022~) : 코인 횡단으로도 유지되는가?
"""
import sqlite3, math, re, json, os, urllib.request
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

DB, KST, COST = "data/ohlcv_data_kst.db", 9 * 3600, 0.0016
MIN_DAYS = 800
SPLIT = pd.Timestamp("2022-01-01")
RSI_TH = 70
con = sqlite3.connect("file:" + DB + "?mode=ro", uri=True)
TODAY_KST = (datetime.now(timezone.utc) + timedelta(hours=9)).date()

tables = [r[0] for r in con.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'coin_KRW_%'")]
tables = sorted(t for t in tables if re.fullmatch(r"coin_KRW_[A-Z0-9]{2,12}", t))
tickers = {t: t.replace("coin_KRW_", "") for t in tables}

# ================================================================ H. 생존편향
print("=" * 92)
print("H. 생존편향 점검   —  상장폐지된 코인이 DB에 남아있는가?")
print("=" * 92)
live = None
try:
    req = urllib.request.Request("https://api.upbit.com/v1/market/all?isDetails=false",
                                 headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        mk = json.loads(r.read().decode())
    live = {m["market"].replace("KRW-", "") for m in mk if m["market"].startswith("KRW-")}
    print("  업비트 현재 원화마켓 : %d개" % len(live))
    print("  DB 보유 코인         : %d개" % len(tickers))
    dead = sorted(set(tickers.values()) - live)
    print("  현재 미상장(폐지 추정): %d개  %s" % (len(dead), ", ".join(dead[:12]) + (" ..." if len(dead) > 12 else "")))
    if len(dead) >= 10:
        print("  => 폐지 코인이 DB에 남아있음. 생존편향은 제한적 (양호)")
    else:
        print("  => 폐지 코인이 거의 없음. 생존편향 위험 (결과를 낙관적으로 볼 것)")
except Exception as e:
    print("  업비트 API 조회 실패(%s) — 생존편향 점검 생략" % type(e).__name__)


# ================================================================ 데이터 로딩
def load(tbl):
    df = pd.read_sql_query(
        "SELECT timestamp, close_price, volume FROM %s WHERE interval_type='day' "
        "ORDER BY timestamp" % tbl, con)
    if len(df) < MIN_DAYS:
        return None
    df.columns = ["ts", "close", "volume"]
    df["date"] = pd.to_datetime(df["ts"] - KST, unit="s", utc=True).dt.tz_localize(None).dt.normalize()
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    if df["date"].iloc[-1].date() >= TODAY_KST:
        df = df.iloc[:-1]
    df = df[(df["close"] > 0)].reset_index(drop=True)
    if len(df) < MIN_DAYS:
        return None
    df["r1"] = df["close"].pct_change()
    dl = df["close"].diff()
    g = dl.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    l = (-dl.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + g / l)
    df["fwd1"] = df["close"].shift(-1) / df["close"] - 1
    return df


print()
print("=" * 92)
print("I. 코인별 RSI>%d 효과   (표본 %d일 이상 코인 전체)" % (RSI_TH, MIN_DAYS))
print("=" * 92)

rows, rets = [], {}
for tbl, tk in tickers.items():
    try:
        df = load(tbl)
    except Exception:
        continue
    if df is None:
        continue
    b = df["fwd1"].dropna()
    s = df.loc[(df["rsi"] > RSI_TH) & df["fwd1"].notna(), "fwd1"]
    if len(s) < 30:
        continue
    seg = df["date"] >= SPLIT
    b2 = df.loc[seg & df["fwd1"].notna(), "fwd1"]
    s2 = df.loc[seg & (df["rsi"] > RSI_TH) & df["fwd1"].notna(), "fwd1"]
    rows.append({
        "ticker": tk, "days": len(df),
        "base_up": (b > 0).mean(), "sig_n": len(s), "sig_up": (s > 0).mean(),
        "diff_up": (s > 0).mean() - (b > 0).mean(),
        "base_ret": b.mean(), "sig_ret": s.mean(), "diff_ret": s.mean() - b.mean(),
        "oos_n": len(s2),
        "oos_diff_up": ((s2 > 0).mean() - (b2 > 0).mean()) if len(s2) >= 20 else np.nan,
        "oos_diff_ret": (s2.mean() - b2.mean()) if len(s2) >= 20 else np.nan,
        "live": (tk in live) if live else None,
    })
    rets[tk] = df.set_index("date")["r1"]

R = pd.DataFrame(rows).sort_values("diff_ret", ascending=False).reset_index(drop=True)
print("  검증 대상 코인: %d개" % len(R))
print()
print("  [상위 8개]                                    [하위 8개]")
print("  %-7s%6s%8s%9s      %-7s%6s%8s%9s"
      % ("티커", "신호일", "상승률차", "수익률차", "티커", "신호일", "상승률차", "수익률차"))
top, bot = R.head(8).reset_index(drop=True), R.tail(8).reset_index(drop=True)
for i in range(8):
    a, z = top.iloc[i], bot.iloc[i]
    print("  %-7s%6d%+8.1f%%p%+9.2f%%      %-7s%6d%+8.1f%%p%+9.2f%%"
          % (a.ticker, a.sig_n, a.diff_up * 100, a.diff_ret * 100,
             z.ticker, z.sig_n, z.diff_up * 100, z.diff_ret * 100))

n = len(R)
pos_up = int((R["diff_up"] > 0).sum())
pos_ret = int((R["diff_ret"] > 0).sum())
print()
print("  " + "-" * 88)
print("  전체 %d개 코인 집계" % n)
print("    상승률이 기준선보다 높은 코인 : %d / %d  (%.1f%%)" % (pos_up, n, pos_up / n * 100))
print("    수익률이 기준선보다 높은 코인 : %d / %d  (%.1f%%)" % (pos_ret, n, pos_ret / n * 100))
print("    상승률 차이  중앙값 %+.2f%%p   평균 %+.2f%%p" % (R["diff_up"].median() * 100, R["diff_up"].mean() * 100))
print("    수익률 차이  중앙값 %+.3f%%    평균 %+.3f%%" % (R["diff_ret"].median() * 100, R["diff_ret"].mean() * 100))
btc = R[R["ticker"] == "BTC"]
if len(btc):
    b = btc.iloc[0]
    rk = int((R["diff_ret"] > b.diff_ret).sum()) + 1
    print("    BTC 위치: 수익률차 %+.3f%% -> %d개 중 %d위 (상위 %.0f%%)"
          % (b.diff_ret * 100, n, rk, rk / n * 100))

# 폐지 코인 분리
if live:
    dead_r = R[R["live"] == False]
    live_r = R[R["live"] == True]
    if len(dead_r) >= 5:
        print()
        print("    [생존편향 분리]")
        print("      현재 상장 %3d개 : 수익률차 중앙값 %+.3f%%   양(+) 비율 %.0f%%"
              % (len(live_r), live_r["diff_ret"].median() * 100, (live_r["diff_ret"] > 0).mean() * 100))
        print("      폐지 추정 %3d개 : 수익률차 중앙값 %+.3f%%   양(+) 비율 %.0f%%"
              % (len(dead_r), dead_r["diff_ret"].median() * 100, (dead_r["diff_ret"] > 0).mean() * 100))
        print("      => 두 그룹이 비슷하면 생존편향 영향 작음")

# ================================================================ J. 독립성
print()
print("=" * 92)
print("J. 독립성 점검   —  알트는 BTC와 같이 움직인다. 실질 독립 표본은 몇 개인가?")
print("=" * 92)
M = pd.DataFrame(rets)
M = M[M.index >= SPLIT].dropna(axis=1, thresh=int(len(M[M.index >= SPLIT]) * 0.8))
C = M.corr()
iu = np.triu_indices_from(C.values, k=1)
avg_corr = np.nanmean(C.values[iu])
k = C.shape[1]
eff = k / (1 + (k - 1) * avg_corr) if avg_corr > 0 else k
print("  2022년 이후 일간수익률 상관 (%d개 코인)" % k)
print("    평균 쌍별 상관계수 : %.3f" % avg_corr)
if "BTC" in C.columns:
    cb = C["BTC"].drop("BTC")
    print("    BTC와의 상관 평균  : %.3f   (최소 %.2f / 최대 %.2f)" % (cb.mean(), cb.min(), cb.max()))
print("    => 실질 독립 표본 수 ≈ %.1f개  (명목 %d개가 아님)" % (eff, k))
print()
print("  * 코인 %d개가 모두 같은 결과를 내도, 사실상 서로 다른 시장 %.0f개 정도의 증거일 뿐"
      % (k, eff))

# ================================================================ K. 검증구간
print()
print("=" * 92)
print("K. 검증구간 (2022~2026)   —  코인 횡단으로도 유지되는가?")
print("=" * 92)
O = R.dropna(subset=["oos_diff_ret"])
print("  검증 가능 코인: %d개 (2022년 이후 신호 20일 이상)" % len(O))
print("    상승률이 기준선보다 높은 코인 : %d / %d  (%.1f%%)"
      % (int((O["oos_diff_up"] > 0).sum()), len(O), (O["oos_diff_up"] > 0).mean() * 100))
print("    수익률이 기준선보다 높은 코인 : %d / %d  (%.1f%%)"
      % (int((O["oos_diff_ret"] > 0).sum()), len(O), (O["oos_diff_ret"] > 0).mean() * 100))
print("    상승률 차이 중앙값 %+.2f%%p    수익률 차이 중앙값 %+.3f%%"
      % (O["oos_diff_up"].median() * 100, O["oos_diff_ret"].median() * 100))

# 코인 전체를 동일가중으로 묶은 포트폴리오 관점
print()
print("  [전 코인 동일가중] 신호일 평균수익 vs 전체일 평균수익  (수수료 %.2f%% 차감 전)" % (COST * 100))
print("    전체일 평균 : %+.3f%%     신호일 평균 : %+.3f%%     차이 %+.3f%%p"
      % (R["base_ret"].mean() * 100, R["sig_ret"].mean() * 100, R["diff_ret"].mean() * 100))
print("    신호일 평균이 수수료(%.2f%%)를 넘는가 : %s"
      % (COST * 100, "예" if R["sig_ret"].mean() > COST else "아니오"))
print("=" * 92)

os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"), exist_ok=True)
R.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "cross_coin.csv"),
         index=False, encoding="utf-8-sig")
print("\n  상세 결과 저장: outputs/cross_coin.csv")
