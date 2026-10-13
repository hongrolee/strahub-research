"""
probe43: 블로그 13편 그림 자료 — probe40 · 41 · 42 와 같은 계산으로 곡선만 다시 뽑음
- curves: 2019-01-01 에 100만 원 → 날마다 가치 (만 원), 7일마다 한 점
  비트코인 들고 있기 · 모든 코인 같은 금액씩 · 규칙 3개(하락 스캘럽 · 헤드앤숄더 · 볼린저 아래, 손절 15 · 익절 50 · 30일) · 100일 평균선 위에서만 비트코인
- ma: 비트코인 종가 · 100일 평균선 (만 원, 7일마다) + 평균선 위였던 구간(날 번호)
- years: probe40 · 42 의 해마다 결과 (규칙 6개 평균은 Hongro 뺌)
결과: outputs/probe43_blog_figures.json
"""
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from probe40_all_in_long import CONDS, load_long, scan_one

HERE = Path(__file__).resolve().parent
FEE = 0.0015
START = '2019-01-01'
RULES = [('desc_scallop', '하락 스캘럽 뒤 사기'), ('hs_top', '헤드앤숄더 천정 뒤 사기'), ('bb_lower', '볼린저 아래로 벗어남')]


def main():
    data = load_long()
    mk = sorted(data)
    days = sorted({r[0] for rows in data.values() for r in rows})
    di = {d: k for k, d in enumerate(days)}
    K, T = len(mk), len(days)
    H, L, C, V = (np.full((K, T), np.nan) for _ in range(4))
    for i, m in enumerate(mk):
        for d, h, l, c, v in data[m]:
            j = di[d]
            H[i, j], L[i, j], C[i, j], V[i, j] = h, l, c, v
    with Pool(6) as pool:
        sig = {}
        for market, rows in pool.imap_unordered(scan_one, [(m, data[m]) for m in mk], chunksize=4):
            i = mk.index(market)
            for cid, day, dr, entry in rows:
                if day in di:
                    sig.setdefault(cid, {}).setdefault(di[day], []).append(i)
    t0, t1 = di[START] if START in di else next(k for k, d in enumerate(days) if d >= START), T - 1

    def run(daysig):
        cash, pos, curve = 1e6, None, []
        for t in range(t0, t1 + 1):
            if pos is None:
                cands = [i for i in daysig.get(t, []) if not np.isnan(C[i, t])]
                if cands:
                    i = max(cands, key=lambda i: V[i, t])
                    pos = (i, t, C[i, t], cash * (1 - FEE) / C[i, t])
                    cash = 0.0
            else:
                i, ts, e, q = pos
                px = None
                if np.isnan(C[i, t]):
                    px = C[i, np.flatnonzero(~np.isnan(C[i, :t]))[-1]]
                elif L[i, t] <= e * 0.85:
                    px = e * 0.85
                elif H[i, t] >= e * 1.5:
                    px = e * 1.5
                elif t - ts >= 30:
                    px = C[i, t]
                if px is not None:
                    cash, pos = q * px * (1 - FEE), None
            curve.append(cash + (pos[3] * C[pos[0], t] if pos else 0))
        return curve

    btc = mk.index('KRW-BTC')
    b = C[btc]
    hold = list(1e6 * (1 - FEE) * b[t0:t1 + 1] / b[t0])
    alive = [i for i in range(K) if not np.isnan(C[i, t0])]
    ew = []
    for t in range(t0, t1 + 1):
        vals = []
        for i in alive:
            k = t if not np.isnan(C[i, t]) else np.flatnonzero(~np.isnan(C[i, :t + 1]))[-1]
            vals.append(C[i, k] / C[i, t0])
        ew.append(1e6 * (1 - FEE) * float(np.mean(vals)))
    ma = np.array([b[max(0, t - 99):t + 1].mean() if t >= 99 else np.nan for t in range(T)])
    cash, qty, filt, above = 1e6, 0.0, [], []
    for t in range(t0, t1 + 1):
        want = b[t - 1] > ma[t - 1]
        if want and qty == 0:
            qty, cash = cash * (1 - FEE) / b[t], 0.0
        elif not want and qty > 0:
            cash, qty = qty * b[t] * (1 - FEE), 0.0
        filt.append(cash + qty * b[t])
        above.append(bool(b[t] > ma[t]))
    curves = {'hold': hold, 'ew': ew, 'ma100': filt}
    for cid, _ in RULES:
        curves[cid] = run(sig.get(cid, {}))
    step = 7
    idx = list(range(0, t1 - t0 + 1, step)) + ([t1 - t0] if (t1 - t0) % step else [])
    spans, run_s = [], None
    for k, on in enumerate(above + [False]):
        if on and run_s is None:
            run_s = k
        elif not on and run_s is not None:
            spans.append([run_s, k - 1])
            run_s = None
    p40 = json.loads((HERE / 'outputs' / 'probe40_all_in_long.json').read_text(encoding='utf-8'))['periods']
    p42 = json.loads((HERE / 'outputs' / 'probe42_btc_trend_core.json').read_text(encoding='utf-8'))
    years = []
    for y in range(2019, int(days[t1][:4]) + 1):
        rows = [r for r in p40[str(y)]['rows'] if not r['name'].startswith('Hongro')]
        years.append({'y': y, 'hold': p42[str(y)]['hold'][0] / 1e4, 'ma100': p42[str(y)]['trend']['100']['final'] / 1e4,
                      'rules': float(np.mean([r['final'] for r in rows])) / 1e4})
    out = {'start': days[t0], 'end': days[t1], 'u': idx,
           'curves': {k: [round(v[u] / 1e4, 3) for u in idx] for k, v in curves.items()},
           'final': {k: round(v[-1]) for k, v in curves.items()},
           'btc': [round(b[t0 + u] / 1e4) for u in idx], 'ma': [round(ma[t0 + u] / 1e4) for u in idx],
           'above': spans, 'years': years}
    (HERE / 'outputs' / 'probe43_blog_figures.json').write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
    print({k: f"{v:,}" for k, v in out['final'].items()})
    print(years)


if __name__ == '__main__':
    main()
