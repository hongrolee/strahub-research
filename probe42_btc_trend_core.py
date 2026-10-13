"""
probe42: 들고 있기를 기본으로 한 가지만 더하기 — 7년(2019-01 ~ 자료 끝) · 해마다 100만 원
① 비트코인 큰 흐름: 종가가 N일 평균선 위면 들고 있고, 아래면 팔고 원화로 쉼 (N = 50 · 100 · 200)
   - 그날 종가로 판단하고 다음 날 종가에 사고팖 (미리 알 수 없게), 사고팔 때마다 0.15%
② 기본 + 시험: 80만 원은 비트코인(한 번에 / 12달 나눠), 20만 원은 probe40 의 규칙 하나 (따로 굴린 두 주머니를 더함)
결과: outputs/probe42_btc_trend_core.json
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DBL = HERE / 'data' / 'ohlcv_data_kst.db'
FEE = 0.0015
KRW = 1_000_000


def main():
    con = sqlite3.connect(f'file:{DBL}?mode=ro', uri=True)
    rows = con.execute('select timestamp, close_price from "coin_KRW_BTC" where interval_type=\'day\' order by timestamp').fetchall()
    days = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts, _ in rows]
    C = np.array([c for _, c in rows], dtype=float)
    T = len(C)
    sma = {n: np.array([C[max(0, t - n + 1):t + 1].mean() if t >= n - 1 else np.nan for t in range(T)]) for n in (50, 100, 200)}

    def trend(n, t0, t1):
        cash, qty, curve, trades, inside = float(KRW), 0.0, [], 0, 0
        for t in range(t0, t1 + 1):
            want = C[t - 1] > sma[n][t - 1]                  # 어제 종가로 정하고 오늘 종가에 사고팖
            if want and qty == 0:
                qty, cash, trades = cash * (1 - FEE) / C[t], 0.0, trades + 1
            elif not want and qty > 0:
                cash, qty, trades = qty * C[t] * (1 - FEE), 0.0, trades + 1
            inside += qty > 0
            curve.append(cash + qty * C[t])
        final = cash + qty * C[t1] * (1 - FEE)
        cv = np.array(curve)
        return final, float(((cv - np.maximum.accumulate(cv)) / np.maximum.accumulate(cv)).min() * 100), trades, inside / (t1 - t0 + 1) * 100

    def hold(t0, t1, parts=1):
        months, buy = [], []
        for t in range(t0, t1 + 1):
            m = days[t][:7]
            if (not months or months[-1] != m) and len(months) < parts:
                months.append(m)
                buy.append(t)
        part = KRW / len(buy)
        cash, qty, curve = float(KRW), 0.0, []
        for t in range(t0, t1 + 1):
            if t in buy:
                qty += part * (1 - FEE) / C[t]
                cash -= part
            curve.append(cash + qty * C[t])
        cv = np.array(curve)
        return cash + qty * C[t1] * (1 - FEE), float(((cv - np.maximum.accumulate(cv)) / np.maximum.accumulate(cv)).min() * 100)

    p40 = json.loads((HERE / 'outputs' / 'probe40_all_in_long.json').read_text(encoding='utf-8'))['periods']
    end = days[-1]
    periods = [('전체', '2019-01-01', end)] + [(str(y), f'{y}-01-01', f'{y}-12-31') for y in range(2019, int(end[:4]) + 1)]
    out = {}
    for pname, a, b in periods:
        t0 = next(k for k, d in enumerate(days) if d >= a)
        t1 = max(k for k, d in enumerate(days) if d <= b)
        h1, hm1 = hold(t0, t1)
        h12, hm12 = hold(t0, t1, 12)
        res = {'from': days[t0], 'to': days[t1], 'hold': [h1, hm1], 'hold12': [h12, hm12], 'trend': {}, 'core': {}}
        print(f"\n=== {pname}: {days[t0]} ~ {days[t1]} · 비트코인 한 번에 {h1:,.0f}원 ({hm1:.0f}%) · 12달 나눠 {h12:,.0f}원 ({hm12:.0f}%)")
        for n in (50, 100, 200):
            f, mdd, tr, ins = trend(n, t0, t1)
            res['trend'][n] = {'final': f, 'mdd': mdd, 'trades': tr, 'inside': ins}
            print(f"  {n:>3}일 평균선 위에서만 들고 있기 → {f:>13,.0f}원 · 가장 크게 줄었을 때 {mdd:6.1f}% · 사고판 횟수 {tr:>3} · 들고 있던 날 {ins:3.0f}%")
        rows40 = p40[pname]['rows']
        for r in rows40:
            core = 0.8 * h1 + 0.2 * r['final']
            core12 = 0.8 * h12 + 0.2 * r['final']
            res['core'][r['name']] = {'lump': core, 'split12': core12, 'rule_only': r['final']}
        best = max(rows40, key=lambda r: r['final'])
        avg = float(np.mean([r['final'] for r in rows40]))
        print(f"  80% 비트코인 한 번에 + 20% 규칙: 규칙 평균이면 {0.8 * h1 + 0.2 * avg:,.0f}원 · 그해 가장 좋은 규칙({best['name']})이면 {0.8 * h1 + 0.2 * best['final']:,.0f}원")
        print(f"  80% 비트코인 12달 나눠 + 20% 규칙: 규칙 평균이면 {0.8 * h12 + 0.2 * avg:,.0f}원")
        res['core_avg'] = {'lump': 0.8 * h1 + 0.2 * avg, 'split12': 0.8 * h12 + 0.2 * avg, 'best_lump': 0.8 * h1 + 0.2 * best['final'], 'best': best['name']}
        out[pname] = res
    (HERE / 'outputs' / 'probe42_btc_trend_core.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
