"""
probe29: 손절을 시장 분위기에 따라 저절로 바꾸면? — 떨어지는 시장이면 손절 10, 오르는 시장이면 20 (probe25 · 28 방식)
- 분위기(그날까지의 값만 씀): 모든 코인 같은 금액 지수 · 비트코인이 N일 평균선 위/아래, 또는 지난 N일 등락 +/−
- 손절 선은 날마다 그날 분위기로 (산 값 기준 −10% 또는 −20%), 익절 50
- 비교: 늘 10 · 15 · 20, 대조(R) = 같은 비율로 아무 날을 '떨어지는 날'로 (분위기 판단에 정보가 있는지)
결과: outputs/probe29_regime_stop.json
"""
import json
from pathlib import Path

import numpy as np

from probe25_bot3_settings import load, spike_prone, COST

HERE = Path(__file__).resolve().parent
TAKE = 50
RNG = np.random.default_rng(5)


def index_series(mk, C):
    r = C[:, 1:] / C[:, :-1] - 1
    with np.errstate(invalid='ignore'):
        m = np.nanmean(np.where(np.isfinite(r), r, np.nan), axis=0)
    idx = np.concatenate([[1.0], np.cumprod(1 + np.nan_to_num(m))])
    btc = C[mk.index('KRW-BTC')].copy()
    for t in range(1, len(btc)):
        if np.isnan(btc[t]):
            btc[t] = btc[t - 1]
    return idx, btc


def down_days(x, kind, n):
    """그날 종가까지 보고 '떨어지는 시장'이면 True (다음 날부터 적용)"""
    out = np.zeros(len(x), bool)
    for t in range(len(x)):
        if t < n:
            continue
        if kind == 'ma':
            out[t] = x[t] < np.mean(x[t - n + 1:t + 1])
        else:
            out[t] = x[t] < x[t - n]
    return out


def sim(e, i, t0, hold, H, L, C, down, lo_s=10, hi_s=20, fixed=None):
    hi = e * (1 + TAKE / 100)
    last = e
    for k in range(1, hold + 1):
        t = t0 + k
        if t >= C.shape[1] or np.isnan(C[i, t]):
            return last, 'gone'
        s = fixed if fixed else (lo_s if down[t - 1] else hi_s)     # 어제 종가까지 본 분위기
        lo = e * (1 - s / 100)
        if L[i, t] <= lo:
            return lo, 'stop'
        if H[i, t] >= hi:
            return hi, 'take'
        last = C[i, t]
    return last, 'end'


def main():
    data = load()
    mk, days, H, L, C, V, first, binance = data
    idx, btc = index_series(mk, C)
    signals = {}
    for src, x in (('시장', idx), ('비트코인', btc)):
        for n in (20, 50):
            signals[f'{src} {n}일 평균선 아래'] = down_days(x, 'ma', n)
        for n in (7, 30):
            signals[f'{src} {n}일 전보다 아래'] = down_days(x, 'ret', n)
    print(f"코인 {len(mk)} · {days[0]} ~ {days[-1]}")
    for k, v in signals.items():
        print(f"  {k}: 떨어지는 날 {v[30:].mean() * 100:.0f}%")
    out = {}
    for hold in (30, 90):
        for pick in ('spike', 'all'):
            starts = list(range(30, len(days) - hold, 7))
            mid = starts[len(starts) // 2]
            pos = [(t0, i) for t0 in starts for i in range(len(mk))
                   if not np.isnan(C[i, t0]) and t0 - first[i] >= 5
                   and (pick == 'all' or spike_prone(i, t0, mk, C, V, first, binance, 0))]

            def score(fn):
                f, b = [], []
                for t0, i in pos:
                    e = C[i, t0]
                    px, _ = fn(e, i, t0)
                    r = (px * (1 - COST)) / (e * (1 + COST)) - 1
                    (f if t0 < mid else b).append(r)
                return {'avg': float(np.mean(f + b) * 100), 'front': float(np.mean(f) * 100), 'back': float(np.mean(b) * 100)}
            rows = []
            for s in (10, 15, 20):
                rows.append({'rule': f'늘 손절 {s}', **score(lambda e, i, t0, s=s: sim(e, i, t0, hold, H, L, C, None, fixed=s))})
            for name, dn in signals.items():
                r = {'rule': f'{name} → 10 / 아니면 20', **score(lambda e, i, t0, dn=dn: sim(e, i, t0, hold, H, L, C, dn))}
                p = dn[30:].mean()
                rnd = RNG.random(len(dn)) < p                                  # 같은 비율의 아무 날
                r['rand'] = score(lambda e, i, t0, rnd=rnd: sim(e, i, t0, hold, H, L, C, rnd))
                rows.append(r)
            out[f'{pick}_{hold}'] = rows
            base = rows[1]
            print(f"\n== {'급등이 잦던 코인' if pick == 'spike' else '모든 코인'} · {hold}일 (앞/뒤 나눔 {days[mid]}) n={len(pos)}")
            for r in rows:
                mark = ' ◎' if r is not base and r['front'] > base['front'] and r['back'] > base['back'] else ''
                line = f"  {r['rule']:<32} 평균 {r['avg']:+6.2f}% | 앞 {r['front']:+6.2f} 뒤 {r['back']:+6.2f}{mark}"
                if 'rand' in r:
                    q = r['rand']
                    line += f"   (아무 날 대조 {q['avg']:+6.2f} → 분위기 몫 {r['avg'] - q['avg']:+5.2f})"
                print(line)
    (HERE / 'outputs' / 'probe29_regime_stop.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
