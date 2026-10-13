"""
probe41: 비트코인(과 이더리움) 사서 들고 있기 — 한 번에 사기 vs 나눠 사기 (매달 1일 같은 금액)
- 자료: data/ohlcv_data_kst.db 일봉 종가 (probe40 과 같음), 2019-01-01 ~ 자료 끝
- 총 100만 원 · 살 때 · 마지막에 팔 때 0.15%씩
- ① 2019-01 부터 끝까지 ② 시작한 해마다 (그해 1월부터 끝까지)
  ③ 아무 달에 시작해 3년 들고 있었다면 (한 번에 vs 12달 나눠) — 손해 본 비율 · 가장 나빴을 때 · 중간값
- 비교: 모든 코인 같은 금액씩 (그날 있던 코인, 이후 없어지면 마지막 값)
결과: outputs/probe41_hold_dca.json
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


def series(con, sym):
    rows = con.execute(f'select timestamp, close_price from "coin_KRW_{sym}" where interval_type=\'day\' order by timestamp').fetchall()
    return {datetime.fromtimestamp(ts).strftime('%Y-%m-%d'): c for ts, c in rows if c}


def main():
    con = sqlite3.connect(f'file:{DBL}?mode=ro', uri=True)
    btc, eth = series(con, 'BTC'), series(con, 'ETH')
    days = sorted(d for d in btc if d >= '2019-01-01')
    end = days[-1]
    months = sorted({d[:7] for d in days})
    first_day = {m: min(d for d in days if d[:7] == m) for m in months}

    def plan(mix, start_m, n_parts, end_day):
        """mix = [(종가, 비중)], start_m 부터 n_parts 달 동안 매달 1일 같은 금액 → end_day 가치 · 가장 크게 줄었을 때"""
        ms = [m for m in months if m >= start_m][:n_parts]
        buy_days = [first_day[m] for m in ms]
        part = KRW / len(buy_days)
        qty = [0.0] * len(mix)
        curve, cash = [], KRW
        for d in days:
            if d < first_day[start_m] or d > end_day:
                continue
            if d in buy_days:
                for k, (s, w) in enumerate(mix):
                    qty[k] += part * w * (1 - FEE) / s[d]
                cash -= part
            curve.append(cash + sum(q * s[d] for q, (s, w) in zip(qty, mix)))
        final = curve[-1] - (curve[-1] - cash) * FEE
        cv = np.array(curve)
        mdd = float(((cv - np.maximum.accumulate(cv)) / np.maximum.accumulate(cv)).min() * 100)
        return final, mdd

    total_months = len(months)
    plans = [
        ('비트코인 · 한 번에', [(btc, 1.0)], 1),
        ('비트코인 · 12달 나눠', [(btc, 1.0)], 12),
        ('비트코인 · 36달 나눠', [(btc, 1.0)], 36),
        ('비트코인 70 + 이더리움 30 · 한 번에', [(btc, .7), (eth, .3)], 1),
        ('비트코인 70 + 이더리움 30 · 12달 나눠', [(btc, .7), (eth, .3)], 12),
        ('이더리움 · 한 번에', [(eth, 1.0)], 1),
    ]

    def P(name, legs, n, start_m, end_day):
        return plan(legs, start_m, n, end_day)

    out = {'end': end, 'all': [], 'by_year': {}, 'rolling': []}
    print(f"① 2019-01-01 ~ {end} (총 100만 원)")
    for name, legs, n in plans:
        f, mdd = P(name, legs, n, '2019-01', end)
        out['all'].append({'name': name, 'final': f, 'mdd': mdd})
        print(f"  {name:<28} → {f:>12,.0f}원 · 가장 크게 줄었을 때 {mdd:6.1f}%")
    n_all = len([m for m in months if m >= '2019-01'])
    f, mdd = P('', [(btc, 1.0)], n_all, '2019-01', end)
    out['all'].append({'name': f'비트코인 · 끝까지 매달 나눠 ({n_all}달)', 'final': f, 'mdd': mdd})
    print(f"  {'비트코인 · 끝까지 매달 나눠':<28} → {f:>12,.0f}원 · 가장 크게 줄었을 때 {mdd:6.1f}%")

    print("\n② 그해 1월에 시작해서 끝까지 들고 있었다면 (한 번에 / 12달 나눠)")
    for y in range(2019, int(end[:4]) + 1):
        sm = f'{y}-01'
        row = {}
        for name, legs, n in plans[:2] + plans[3:5]:
            row[name] = P(name, legs, n, sm, end)
        out['by_year'][y] = {k: {'final': v[0], 'mdd': v[1]} for k, v in row.items()}
        print(f"  {y}: " + ' · '.join(f"{k} {v[0]:,.0f}원 ({v[1]:.0f}%)" for k, v in row.items()))

    print("\n③ 아무 달에 시작해서 3년 들고 있었다면 (비트코인)")
    starts = [m for m in months if m >= '2019-01']
    res = {1: [], 12: []}
    for sm in starts:
        k = months.index(sm) + 36
        if k >= len(months):
            break
        end_day = first_day[months[k]]
        for n in (1, 12):
            res[n].append((sm, *plan([(btc, 1.0)], sm, n, end_day)))
    for n, rows in res.items():
        fs = np.array([r[1] for r in rows])
        worst = min(rows, key=lambda r: r[1])
        best = max(rows, key=lambda r: r[1])
        d = {'n': n, 'starts': len(rows), 'loss_pct': float((fs < KRW).mean() * 100), 'median': float(np.median(fs)),
             'worst': [worst[0], worst[1]], 'best': [best[0], best[1]], 'mdd_median': float(np.median([r[2] for r in rows]))}
        out['rolling'].append(d)
        print(f"  {'한 번에' if n == 1 else '12달 나눠'}: 시작한 달 {len(rows)}개 · 3년 뒤 손해 {d['loss_pct']:.0f}% · 중간값 {d['median']:,.0f}원"
              f" · 가장 나쁨 {worst[0]} 시작 {worst[1]:,.0f}원 · 가장 좋음 {best[0]} 시작 {best[1]:,.0f}원 · 줄었을 때 중간값 {d['mdd_median']:.0f}%")
    (HERE / 'outputs' / 'probe41_hold_dca.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
