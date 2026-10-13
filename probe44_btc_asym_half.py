"""
probe44: 비트코인 평균선 쉬기 — 오르는 해에도 덜 놓치는 방법 (7년 · 해마다)
① 대칭: 어제 종가 > N일 평균이면 들고, 아래면 팖 (probe42 와 같음)
② 비대칭: 평균선 위로 b% 넘으면 사고, 아래로 s% 내려가야 팖 (사기는 쉽게, 팔기는 느슨하게)
③ 반반: 50% 그냥 들고 있기 + 50% 평균선 규칙 (두 주머니를 따로 굴려 더함)
- 자료: data/ohlcv_data_kst.db 비트코인 일봉, 2019-01-01 ~ 끝, 100만 원, 사고팔 때마다 0.15%
- 어제 종가로 정하고 오늘 종가에 사고팖 (미리 알 수 없게)
결과: outputs/probe44_btc_asym_half.json
"""
import itertools
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FEE = 0.0015
KRW = 1_000_000


def main():
    con = sqlite3.connect(f"file:{HERE / 'data' / 'ohlcv_data_kst.db'}?mode=ro", uri=True)
    rows = con.execute('select timestamp, close_price from "coin_KRW_BTC" where interval_type=\'day\' order by timestamp').fetchall()
    days = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts, _ in rows]
    C = np.array([c for _, c in rows], dtype=float)
    T = len(C)
    sma = {n: np.array([C[max(0, t - n + 1):t + 1].mean() if t >= n - 1 else np.nan for t in range(T)]) for n in (50, 100, 200)}

    def rule(n, b, s, t0, t1):
        """b · s = %; 위로 b% 넘으면 사고 아래로 s% 내려가면 팖 (그 사이는 지금 상태 그대로)"""
        cash, qty, curve, trades = float(KRW), 0.0, [], 0
        for t in range(t0, t1 + 1):
            ratio = C[t - 1] / sma[n][t - 1] - 1
            if qty == 0 and ratio > b / 100:
                qty, cash, trades = cash * (1 - FEE) / C[t], 0.0, trades + 1
            elif qty > 0 and ratio < -s / 100:
                cash, qty, trades = qty * C[t] * (1 - FEE), 0.0, trades + 1
            curve.append(cash + qty * C[t])
        curve[-1] = cash + qty * C[t1] * (1 - FEE)
        return np.array(curve), trades

    def hold(t0, t1):
        return np.array([KRW * (1 - FEE) * C[t] / C[t0] for t in range(t0, t1 + 1)]) * np.r_[np.ones(t1 - t0), 1 - FEE]

    def mdd(cv):
        return float(((cv - np.maximum.accumulate(cv)) / np.maximum.accumulate(cv)).min() * 100)

    end = days[-1]
    periods = [('전체', '2019-01-01', end)] + [(str(y), f'{y}-01-01', f'{y}-12-31') for y in range(2019, int(end[:4]) + 1)]
    idx = {p: (next(k for k, d in enumerate(days) if d >= a), max(k for k, d in enumerate(days) if d <= b)) for p, a, b in periods}

    grid = list(itertools.product((50, 100, 200), (0, 0.5, 1, 2), (0, 2, 3, 5, 8, 10)))
    res = {}
    for n, b, s in grid:
        for p, (t0, t1) in idx.items():
            cv, tr = rule(n, b, s, t0, t1)
            h = hold(t0, t1)
            half = (cv + h) / 2
            res.setdefault((n, b, s), {})[p] = {'rule': cv[-1], 'half': half[-1], 'mdd': mdd(cv), 'half_mdd': mdd(half), 'trades': tr}
    hold_res = {p: {'final': hold(*idx[p])[-1], 'mdd': mdd(hold(*idx[p]))} for p in idx}
    years = [p for p in idx if p != '전체']

    def show(name, f, key='rule'):
        r = res[f]
        print(f"  {name:<34} 7년 {r['전체'][key] / 1e4:>8,.0f}만 · 줄었을 때 {r['전체']['mdd' if key == 'rule' else 'half_mdd']:6.1f}% · 거래 {r['전체']['trades']:>3} | "
              + ' '.join(f"{y[2:]}:{r[y][key] / 1e4:>4.0f}" for y in years))

    print(f"비트코인 일봉 {days[idx['전체'][0]]} ~ {end}, 100만 원 (단위: 만 원, 해마다는 그해 1월 1일 100만 원으로 새로)")
    print(f"  {'그냥 들고 있기':<34} 7년 {hold_res['전체']['final'] / 1e4:>8,.0f}만 · 줄었을 때 {hold_res['전체']['mdd']:6.1f}% |         "
          + ' '.join(f"{y[2:]}:{hold_res[y]['final'] / 1e4:>4.0f}" for y in years))
    for n in (50, 100, 200):
        show(f'① 대칭 {n}일', (n, 0, 0))
    # 오르는 해(그냥 들고 있기가 이긴 해)와 내리는 해 나눠서 평균
    up = [y for y in years if hold_res[y]['final'] > KRW * 1.05]
    down = [y for y in years if y not in up]
    print('오르는 해', up, '· 내리는 해', down)

    def score(f, key='rule'):
        r = res[f]
        return (np.mean([r[y][key] for y in up]) / 1e4, np.mean([r[y][key] for y in down]) / 1e4, r['전체'][key] / 1e4)
    print(f"  {'그냥 들고 있기':<34} 오르는 해 평균 {np.mean([hold_res[y]['final'] for y in up]) / 1e4:5.0f}만 · 내리는 해 평균 {np.mean([hold_res[y]['final'] for y in down]) / 1e4:5.0f}만")
    rows = []
    for f in grid:
        for key in ('rule', 'half'):
            u, d, a = score(f, key)
            rows.append((f, key, u, d, a))
    print('\n② 비대칭 · ③ 반반: 오르는 해 평균이 높은 순 (내리는 해 · 7년 함께)')
    for f, key, u, d, a in sorted(rows, key=lambda x: -x[2])[:12]:
        print(f"  {'반반 ' if key == 'half' else ''}{f[0]}일 · 사기 +{f[1]}% · 팔기 −{f[2]}%".ljust(32), f"오르는 해 {u:5.0f}만 · 내리는 해 {d:5.0f}만 · 7년 {a:7,.0f}만 · 줄었을 때 {res[f]['전체']['mdd' if key == 'rule' else 'half_mdd']:6.1f}%")
    print('\n두 쪽 다 고려: (오르는 해 + 내리는 해) 평균이 높은 순')
    for f, key, u, d, a in sorted(rows, key=lambda x: -(x[2] + x[3]))[:12]:
        print(f"  {'반반 ' if key == 'half' else ''}{f[0]}일 · 사기 +{f[1]}% · 팔기 −{f[2]}%".ljust(32), f"오르는 해 {u:5.0f}만 · 내리는 해 {d:5.0f}만 · 7년 {a:7,.0f}만 · 줄었을 때 {res[f]['전체']['mdd' if key == 'rule' else 'half_mdd']:6.1f}%")
    print('\n자세히 볼 것')
    for name, f, key in [('① 대칭 100일', (100, 0, 0), 'rule'), ('② 비대칭 100일 · +0.5 / −5', (100, 0.5, 5), 'rule'),
                         ('② 비대칭 100일 · +0 / −8', (100, 0, 8), 'rule'), ('② 비대칭 200일 · +0 / −5', (200, 0, 5), 'rule'),
                         ('③ 반반 (대칭 100일)', (100, 0, 0), 'half'), ('③ 반반 (100일 · +0 / −5)', (100, 0, 5), 'half')]:
        show(name, f, key)
    out = {'days': [days[idx['전체'][0]], end], 'up': up, 'down': down, 'hold': hold_res,
           'res': [{'n': f[0], 'b': f[1], 's': f[2], **{p: v for p, v in d.items()}} for f, d in res.items()]}
    (HERE / 'outputs' / 'probe44_btc_asym_half.json').write_text(json.dumps(out, ensure_ascii=False, default=float), encoding='utf-8')


if __name__ == '__main__':
    main()
