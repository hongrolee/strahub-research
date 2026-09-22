# -*- coding: utf-8 -*-
"""
BTC 방향 예측 가능성 탐색 (측정 전용, 읽기 전용 DB)
  1) 구간별(1/3/7/14/30일) 방향 적중률  -> "내일"이 정말 최악의 구간인가
  2) 조건부 적중률                      -> 특정 상황에서만 예측력이 있는가
  3) 손익비/기대수익 (수수료 반영)      -> 적중률이 아니라 돈이 되는가
"""
import sqlite3, math
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

DB = "data/ohlcv_data_kst.db"
TABLE = "coin_KRW_BTC"
KST = 9 * 3600
COST = 0.0016          # 왕복 수수료+슬리피지 0.16%
rng = np.random.default_rng(42)


# ---------------------------------------------------------------- 데이터
def load():
    con = sqlite3.connect("file:" + DB + "?mode=ro", uri=True)
    q = ("SELECT timestamp, open_price, high_price, low_price, close_price, volume "
         "FROM " + TABLE + " WHERE interval_type='day' ORDER BY timestamp")
    df = pd.read_sql_query(q, con)
    con.close()
    df.columns = ["ts", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["ts"] - KST, unit="s", utc=True).dt.tz_localize(None).dt.normalize()
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)

    # 미완성 캔들 제거 (자동 갱신 DB라 실행 시점마다 발생)
    dropped = []
    today_kst = (datetime.now(timezone.utc) + timedelta(hours=9)).date()
    if df["date"].iloc[-1].date() >= today_kst:
        dropped.append(str(df["date"].iloc[-1].date()) + " (오늘 날짜, 장중)")
        df = df.iloc[:-1]
    med = df["volume"].iloc[-31:-1].median()
    if df["volume"].iloc[-1] < 0.2 * med:
        dropped.append("%s (거래량 %.1f < 최근중앙값 %.1f 의 20%%)"
                       % (df["date"].iloc[-1].date(), df["volume"].iloc[-1], med))
        df = df.iloc[:-1]
    return df.reset_index(drop=True), dropped


def integrity(df):
    gaps = int((df["date"].diff().dt.days.dropna() != 1).sum())
    bad = int(((df["low"] > df[["open", "close"]].min(axis=1)) |
               (df["high"] < df[["open", "close"]].max(axis=1))).sum())
    return gaps, bad


# ---------------------------------------------------------------- 통계 도구
def prop_ci(k, n, z=1.96):
    """Wilson score 구간 (소표본에서 정규근사보다 안정적)"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def boot_mean_ci(x, n_boot=2000, block=5):
    """블록 부트스트랩 평균 신뢰구간 (자기상관 고려), 벡터화"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 10:
        return (float("nan"), float("nan"))
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, max(n - block + 1, 1), size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = x[np.minimum(idx, n - 1)].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


# ---------------------------------------------------------------- 0. 점검
df, dropped = load()
gaps, bad = integrity(df)
c = df["close"].values
n = len(df)

print("=" * 80)
print("0. 데이터 점검")
print("=" * 80)
print("  기간      : %s ~ %s  (%d행)" % (df["date"].iloc[0].date(), df["date"].iloc[-1].date(), n))
print("  날짜 gap  : %d건     OHLC 정합 위반: %d건" % (gaps, bad))
print("  제외      : %s" % ("  /  ".join(dropped) if dropped else "없음"))
_r = pd.Series(c).pct_change()
print("  일간수익률: 평균 %+.3f%%   표준편차 %.2f%%" % (_r.mean() * 100, _r.std() * 100))

# ---------------------------------------------------------------- 1. 구간별
print()
print("=" * 80)
print("1. 예측 구간별 방향 적중률   ('내일'이 정말 최악인가?)")
print("=" * 80)
print("%6s %7s %9s %9s %19s %10s" % ("구간", "표본", "상승비율", "관성추종", "95%CI(관성)", "판정"))
print("-" * 80)
for h in [1, 3, 7, 14, 30]:
    fwd = c[h:] / c[:-h] - 1                   # t -> t+h 수익률
    up = fwd > 0
    prev, nxt = up[:-h], up[h:]                # 직전 구간 방향 -> 다음 구간 방향
    idx = np.arange(0, len(prev), h)           # 비중첩 표본만 사용
    hit = (prev[idx] == nxt[idx])
    k, m = int(hit.sum()), len(hit)
    lo, hi = prop_ci(k, m)
    verdict = "유의" if lo > 0.5 else ("역신호" if hi < 0.5 else "무의미")
    print("%4d일 %7d %8.1f%% %8.1f%%   [%5.1f%%, %5.1f%%] %10s"
          % (h, m, up.mean() * 100, k / m * 100, lo * 100, hi * 100, verdict))
print()
print("  * 관성추종 = '직전 구간에 올랐으면 다음 구간도 오른다'")
print("  * 95%CI가 50%를 포함하면 동전던지기와 구분 불가")

# ---------------------------------------------------------------- 2. 조건부
print()
print("=" * 80)
print("2. 조건부 적중률   (특정 상황에서만 예측력이 있는가?)")
print("=" * 80)

d = df.copy()
d["r1"] = d["close"].pct_change()
d["sig20"] = d["r1"].rolling(20).std()

delta = d["close"].diff()                       # RSI(14), Wilder
gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
d["rsi"] = 100 - 100 / (1 + gain / loss)

ma20 = d["close"].rolling(20).mean()
sd20 = d["close"].rolling(20).std()
d["bb_pctb"] = (d["close"] - (ma20 - 2 * sd20)) / (4 * sd20)
d["vol_pct"] = d["sig20"].rolling(252).rank(pct=True)   # 최근 252일 내 변동성 백분위
d["down_streak"] = (d["r1"] < 0).astype(int).groupby((d["r1"] >= 0).cumsum()).cumsum()
d["up_streak"] = (d["r1"] > 0).astype(int).groupby((d["r1"] <= 0).cumsum()).cumsum()

for h in [1, 3, 7]:                             # 예측 대상: t종가 -> t+h종가
    d["fwd%d" % h] = d["close"].shift(-h) / d["close"] - 1

CONDS = {
    "패닉 급락 (r1 < -7%)": d["r1"] < -0.07,
    "급등 (r1 > +7%)": d["r1"] > 0.07,
    "3일 연속 하락": d["down_streak"] >= 3,
    "5일 연속 하락": d["down_streak"] >= 5,
    "3일 연속 상승": d["up_streak"] >= 3,
    "변동성 최저 10% 구간": d["vol_pct"] < 0.10,
    "변동성 최고 10% 구간": d["vol_pct"] > 0.90,
    "RSI < 30 (과매도)": d["rsi"] < 30,
    "RSI > 70 (과매수)": d["rsi"] > 70,
    "볼린저 하단 이탈 (%b<0)": d["bb_pctb"] < 0,
    "볼린저 상단 이탈 (%b>1)": d["bb_pctb"] > 1,
}

base = d["fwd1"].dropna()
base_up = (base > 0).mean()
print("  [기준선] 전체 %d일 중 다음날 상승 %.2f%%   평균 %+.3f%%"
      % (len(base), base_up * 100, base.mean() * 100))
print()
print("%-26s%6s%8s%18s%10s%9s" % ("조건", "표본", "상승률", "95%CI", "평균수익", "판정"))
print("-" * 80)
for name, mask in CONDS.items():
    sub = d.loc[mask & d["fwd1"].notna(), "fwd1"]
    m = len(sub)
    if m == 0:
        print("%-26s%6d  (해당 없음)" % (name, 0))
        continue
    k = int((sub > 0).sum())
    lo, hi = prop_ci(k, m)
    if lo > base_up:
        verdict = "상승우위"
    elif hi < base_up:
        verdict = "하락우위"
    else:
        verdict = "무의미"
    if m < 30:
        verdict += "*"
    print("%-26s%6d%7.1f%%  [%5.1f%%,%5.1f%%]%+9.2f%%%9s"
          % (name, m, k / m * 100, lo * 100, hi * 100, sub.mean() * 100, verdict))
print()
print("  * 비교 기준은 50%%가 아니라 기준선 %.1f%% (BTC는 상방편향이 있음)" % (base_up * 100))
print("  * 표본 30 미만은 * 표시 — 숫자가 좋아 보여도 신뢰 불가")
print("  * 조건 %d개 동시 검정 -> Bonferroni 보정 시 유의수준 %.4f 필요"
      % (len(CONDS), 0.05 / len(CONDS)))

# ---------------------------------------------------------------- 3. 손익비
print()
print("=" * 80)
print("3. 손익비 / 기대수익   (적중률이 아니라 돈이 되는가?)   왕복비용 0.16% 반영")
print("=" * 80)
print("%-25s%5s%7s%8s%8s%7s%9s%19s"
      % ("조건 / 보유기간", "표본", "승률", "평균익", "평균손", "손익비", "순기대값", "95%CI(순기대값)"))
print("-" * 88)


def payoff(sub, label):
    win, lose = sub[sub > 0], sub[sub <= 0]
    if len(win) == 0 or len(lose) == 0:
        return
    avg_w, avg_l = win.mean(), lose.mean()
    net = sub - COST
    lo, hi = boot_mean_ci(net.values)
    flag = "  <== 양수" if lo > 0 else ""
    print("%-25s%5d%6.1f%%%+7.2f%%%+7.2f%%%7.2f%+8.2f%%  [%+6.2f%%,%+6.2f%%]%s"
          % (label, len(sub), (sub > 0).mean() * 100, avg_w * 100, avg_l * 100,
             avg_w / abs(avg_l), net.mean() * 100, lo * 100, hi * 100, flag))


payoff(d["fwd1"].dropna(), "[기준] 매일 롱 / 1일")
print("-" * 88)
for name, mask in CONDS.items():
    for hcol, hn in [("fwd1", "1일"), ("fwd7", "7일")]:
        sub = d.loc[mask & d[hcol].notna(), hcol]
        if len(sub) >= 15:
            payoff(sub, "%s / %s" % (name[:18], hn))

print()
print("  * 순기대값 = 평균수익 - 0.16%(왕복비용).  95%CI 하한이 0보다 커야 의미 있음")
print("  * 7일 보유는 구간이 겹쳐 표본이 부풀려짐 -> CI가 실제보다 좁게 나옴 (주의)")
print("=" * 80)
