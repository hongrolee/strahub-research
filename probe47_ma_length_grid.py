"""
probe47: 평균선 길이 · 파는 기준 찾기 — 단순 평균(SMA) · 지수 평균(EMA) × 기간 × '아래 몇 %에 팖'
- 규칙: 어제 종가가 평균선 위면 사고(0%), 평균선보다 s% 넘게 아래면 팖 · 일봉 · 오늘 종가에 사고팖 · 사고팔 때마다 0.15%
- 자료: data/ohlcv_data_kst.db 일봉, 코인 229개(지금 상장, 스테이블 뺌), 2019-01-01 (또는 상장 210일 뒤) ~ 끝
- 숫자에 맞춘 우연을 걸러내려고 앞 기간(2019~2022)에서 고른 숫자가 뒤 기간(2023~2026)에도 좋은지 따로 봄
결과: outputs/probe47_ma_length_grid.json
"""
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from probe45_ma50_band_all_coins import load, FEE

HERE = Path(__file__).resolve().parent
LENS = [20, 30, 40, 50, 60, 80, 100, 120, 150, 200]
BANDS = [0, 1, 2, 3, 5]
KINDS = ['sma', 'ema']
SPLIT = '2023-01-01'


def ma_of(C, kind, n):
    if kind == 'ema':
        out = np.empty(len(C))
        out[0] = C[0]
        k = 2 / (n + 1)
        for i in range(1, len(C)):
            out[i] = C[i] * k + out[i - 1] * (1 - k)
    else:
        out = np.convolve(C, np.ones(n) / n, mode='full')[:len(C)]
    out[:n - 1] = np.nan
    return out


def run(C, ma, s, t0, t1):
    cash, qty, peak, mdd, trades = 1.0, 0.0, 1.0, 0.0, 0
    for t in range(t0, t1 + 1):
        m = ma[t - 1]
        if m == m:
            r = C[t - 1] / m - 1
            if qty == 0 and r > 0:
                qty, cash, trades = cash * (1 - FEE) / C[t], 0.0, trades + 1
            elif qty > 0 and r < -s / 100:
                cash, qty, trades = qty * C[t] * (1 - FEE), 0.0, trades + 1
        v = cash + qty * C[t]
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    return cash + qty * C[t1] * (1 - FEE), mdd * 100, trades


def one_coin(args):
    sym, rows = args
    days = [d for d, _ in rows]
    C = np.array([c for _, c in rows], dtype=float)
    t0 = max(next((k for k, d in enumerate(days) if d >= '2019-01-01'), len(days)), 210)
    t1 = len(days) - 1
    if t0 >= t1 - 60:
        return sym, None
    ts = next((k for k, d in enumerate(days) if d >= SPLIT), None)
    out = {'from': days[t0], 'hold': C[t1] / C[t0] * (1 - FEE) ** 2}
    parts = {'all': (t0, t1)}
    if ts and ts - t0 > 120:
        parts['front'] = (t0, ts - 1)
        out['hold_front'] = C[ts - 1] / C[t0] * (1 - FEE) ** 2
    if ts and t1 - max(ts, t0) > 120:
        parts['back'] = (max(ts, t0), t1)
        out['hold_back'] = C[t1] / C[max(ts, t0)] * (1 - FEE) ** 2
    res = {}
    for kind in KINDS:
        for n in LENS:
            ma = ma_of(C, kind, n)
            for s in BANDS:
                res[f'{kind}{n}/{s}'] = {p: run(C, ma, s, a, b) for p, (a, b) in parts.items()}
    out['res'] = res
    return sym, out


def main():
    data = load()
    with Pool(8) as pool:
        allres = {s: r for s, r in pool.imap_unordered(one_coin, list(data.items()), chunksize=4) if r}
    coins = sorted(allres)
    keys = list(allres[coins[0]]['res'])
    print(f"코인 {len(coins)}개 · 조합 {len(keys)}가지 (평균 2가지 × 길이 {len(LENS)} × 파는 기준 {len(BANDS)})")

    def stat(key, part='all', only=None):
        cs = [c for c in (only or coins) if part in allres[c]['res'][key]]
        f = np.array([allres[c]['res'][key][part][0] for c in cs])
        h = np.array([allres[c]['hold' if part == 'all' else f'hold_{part}'] for c in cs])
        dd = np.array([allres[c]['res'][key][part][1] for c in cs])
        tr = np.array([allres[c]['res'][key][part][2] for c in cs])
        return {'n': len(cs), 'median': float(np.median(f)), 'profit': float((f > 1).mean() * 100), 'beat': float((f > h).mean() * 100),
                'mdd': float(np.median(dd)), 'trades': float(np.median(tr))}

    table = {k: {'all': stat(k), 'front': stat(k, 'front'), 'back': stat(k, 'back')} for k in keys}
    btc = allres['BTC']['res']

    def line(k):
        a, f, b = table[k]['all'], table[k]['front'], table[k]['back']
        bt = btc[k]['all']
        return (f"{k:<10} 모든 코인 가운데 {a['median']:5.2f}배 · 번 코인 {a['profit']:3.0f}% · 들고보다 나음 {a['beat']:3.0f}% · 줄었을 때 {a['mdd']:4.0f}% · 거래 {a['trades']:3.0f}"
                f" | 앞 {f['median']:4.2f} · 뒤 {b['median']:4.2f} | BTC {bt[0]:6.1f}배 ({bt[1]:.0f}%, {bt[2]}번)")

    print('\n=== 모든 코인 가운데 값 큰 순 10')
    for k in sorted(keys, key=lambda k: -table[k]['all']['median'])[:10]:
        print(' ', line(k))
    print('\n=== BTC 큰 순 10')
    for k in sorted(keys, key=lambda k: -btc[k]['all'][0])[:10]:
        print(' ', line(k))
    print('\n=== 앞 기간(2019~2022) 1~5등이 뒤 기간(2023~)에도 좋은가 — 뒤 기간 순위 (전체 100가지 중)')
    back_rank = {k: i + 1 for i, k in enumerate(sorted(keys, key=lambda k: -table[k]['back']['median']))}
    for k in sorted(keys, key=lambda k: -table[k]['front']['median'])[:5]:
        print(f"  {k:<10} 앞 {table[k]['front']['median']:.2f}배 → 뒤 {table[k]['back']['median']:.2f}배 (뒤 기간 {back_rank[k]}등)")
    print('\n=== 길이별 (파는 기준 2% 고정) — 모든 코인 가운데 · BTC')
    for kind in KINDS:
        print('  ' + kind.upper() + ': ' + ' · '.join(f"{n}:{table[f'{kind}{n}/2']['all']['median']:.2f}/{btc[f'{kind}{n}/2']['all'][0]:.0f}" for n in LENS))
    print('\n=== 파는 기준별 (SMA50 · EMA120) — 모든 코인 가운데 · BTC')
    for k0 in ('sma50', 'ema120'):
        print(f"  {k0}: " + ' · '.join(f"{s}%:{table[f'{k0}/{s}']['all']['median']:.2f}/{btc[f'{k0}/{s}']['all'][0]:.0f}" for s in BANDS))
    print('\n=== 지금 쓰는 숫자')
    for k in ('sma50/2', 'ema120/2'):
        print(' ', line(k), f"| 모든 코인 순위 {sorted(keys, key=lambda x: -table[x]['all']['median']).index(k) + 1} · BTC 순위 {sorted(keys, key=lambda x: -btc[x]['all'][0]).index(k) + 1}")
    (HERE / 'outputs' / 'probe47_ma_length_grid.json').write_text(json.dumps({'table': table, 'btc': {k: v for k, v in btc.items()}}, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
