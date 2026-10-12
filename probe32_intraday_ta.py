"""
probe32: 짧은 봉에서 트레이딩뷰 인기 지표 — Supertrend(10·3) · Squeeze Momentum[LazyBear] · SMC[LuxAlgo] 구조(내부 5 · 큰 흐름 50)
- 자료 ① 15분봉 1년 74종 (data/box_cache)  ② 같은 15분봉을 1시간봉으로  ③ 1분봉 약 70일 288종(data/spike_cache_all)을 1시간봉으로
- 신호 봉 종가에 산 뒤 4시간 · 1일 · 3일 뒤 수익(비용 한쪽 0.15%) vs 같은 코인 앞뒤 1일 안 아무 봉에 산 것
  사는 신호: 그 차이가 + 여야 정보가 있음 / 파는 신호: 신호 뒤가 아무 봉보다 더 내려야(− 여야) 정보가 있음
- 앞 절반 · 뒤 절반 기간을 따로 봐서 둘 다 같은 쪽이어야 믿을 만함
결과: outputs/probe32_intraday_ta.json
"""
import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as win

from probe30_supertrend import supertrend
from probe31_squeeze_smc import structure

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(32)
COST = 0.0015


def load_15m():
    out = {}
    for f in sorted((HERE / 'data' / 'box_cache').glob('KRW-*_15m_365d_*.json')):
        rows = json.loads(f.read_text(encoding='utf-8'))
        a = np.array([[r['o'], r['h'], r['l'], r['c'], r.get('v') or 0.0] for r in rows], float)
        t = np.array([np.datetime64(r['t']).astype('datetime64[s]').astype(np.int64) for r in rows])
        out[f.name.split('_')[0]] = (t, a)
    return out


def to_hour(t, a, sec):
    """봉 묶기 (sec 초 단위): 시가 · 고가 · 저가 · 종가 · 거래대금"""
    k = t // sec
    cut = np.flatnonzero(np.diff(k)) + 1
    idx = np.split(np.arange(len(t)), cut)
    o = np.array([a[i[0], 0] for i in idx]); h = np.array([a[i, 1].max() for i in idx]); l = np.array([a[i, 2].min() for i in idx])
    c = np.array([a[i[-1], 3] for i in idx]); v = np.array([a[i, 4].sum() for i in idx])
    return k[[i[0] for i in idx]] * sec, np.stack([o, h, l, c, v], 1)


def load_1m_hour():
    out = {}
    for f in sorted((HERE / 'data' / 'spike_cache_all').glob('KRW-*.npz')):
        z = np.load(f)
        t = z['t'].astype(np.int64)
        a = np.stack([z['o'], z['h'], z['l'], z['c'], z['v']], 1).astype(float)
        out[f.stem] = to_hour(t, a, 3600)
    return out


def rolling_mean(x, n):
    o = np.full(len(x), np.nan)
    if len(x) >= n:
        o[n - 1:] = win(x, n).mean(1)
    return o


def squeeze_fast(h, l, c, n=20, bb=2.0, kc=1.5):
    basis = rolling_mean(c, n)
    sd = np.full(len(c), np.nan)
    sd[n - 1:] = win(c, n).std(1)
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.fmax(h - l, np.fmax(np.abs(h - pc), np.abs(l - pc)))
    rng = rolling_mean(tr, n)
    on = (basis - bb * sd > basis - rng * kc) & (basis + bb * sd < basis + rng * kc)
    off = (basis - bb * sd < basis - rng * kc) & (basis + bb * sd > basis + rng * kc)
    hh = np.full(len(c), np.nan); ll = hh.copy()
    hh[n - 1:] = win(h, n).max(1); ll[n - 1:] = win(l, n).min(1)
    src = c - ((hh + ll) / 2 + basis) / 2
    val = np.full(len(c), np.nan)
    x = np.arange(n); xm = x.mean(); den = ((x - xm) ** 2).sum()
    ok = np.flatnonzero(~np.isnan(src))
    if len(ok) >= n:
        s0 = ok[0]
        w = win(src[s0:], n)
        b = ((w - w.mean(1, keepdims=True)) * (x - xm)).sum(1) / den
        val[s0 + n - 1:] = w.mean(1) + b * (n - 1 - xm)
    return on, off, val


def signals(a):
    o, h, l, c = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    D = supertrend(h[None], l[None], c[None], 10, 3.0)[0]
    on, off, val = squeeze_fast(h, l, c)
    pon = np.concatenate([[False], on[:-1]])
    pv = np.concatenate([[np.nan], val[:-1]]); ppv = np.concatenate([[np.nan], pv[:-1]])
    fired = off & pon
    iub, iuc, idb, idc = structure(h, l, c, 5)
    sub, suc, sdb, sdc = structure(h, l, c, 50)
    pD = np.concatenate([[0], D[:-1]])
    with np.errstate(invalid='ignore'):
        buy = {
            'ST 오름으로 바뀜': (D == 1) & (pD == -1),
            'SQ 터짐 & 모멘텀 +': fired & (val > 0),
            'SQ 모멘텀 0 위로': (val > 0) & (pv <= 0),
            'SQ 바닥에서 돌아섬 (− 인데 오르기 시작)': (val < 0) & (val > pv) & (ppv >= pv),
            'SMC 위로 CHoCH (내부 5)': iuc, 'SMC 위로 BOS (내부 5)': iub,
            'SMC 위로 CHoCH (큰 흐름 50)': suc, 'SMC 위로 BOS (큰 흐름 50)': sub,
        }
        sell = {
            'ST 내림으로 바뀜': (D == -1) & (pD == 1),
            'SQ 터짐 & 모멘텀 −': fired & (val < 0),
            'SQ 모멘텀 0 아래로': (val < 0) & (pv >= 0),
            'SMC 아래로 CHoCH (내부 5)': idc, 'SMC 아래로 BOS (내부 5)': idb,
            'SMC 아래로 CHoCH (큰 흐름 50)': sdc, 'SMC 아래로 BOS (큰 흐름 50)': sdb,
        }
    return buy, sell


def evaluate(series, bars_per_hour):
    """series: {coin: (t, a)} → {신호: {보유: {n, sig, rand, edge, f, b}}}"""
    holds = {'4시간': 4 * bars_per_hour, '1일': 24 * bars_per_hour, '3일': 72 * bars_per_hour}
    near = 24 * bars_per_hour
    acc = {}
    tmin = min(t[0] for t, _ in series.values()); tmax = max(t[-1] for t, _ in series.values())
    mid = (tmin + tmax) // 2
    for coin, (t, a) in series.items():
        if len(t) < 300:
            continue
        c = a[:, 3]
        buy, sell = signals(a)
        for side, group in (('buy', buy), ('sell', sell)):
            for name, S in group.items():
                idx = np.flatnonzero(S)
                idx = idx[(idx >= 60) & (idx < len(c) - max(holds.values()) - near - 1)]
                for hname, H in holds.items():
                    rec = acc.setdefault((side, name, hname), {'sig': [], 'rand': [], 'half': []})
                    for i in idx:
                        u = int(np.clip(i + RNG.integers(-near, near + 1), 60, len(c) - H - 1))
                        rec['sig'].append(c[i + H] / c[i] - 1 - 2 * COST)
                        rec['rand'].append(c[u + H] / c[u] - 1 - 2 * COST)
                        rec['half'].append(t[i] >= mid)
    out = {}
    for (side, name, hname), r in acc.items():
        s, q, half = np.array(r['sig']), np.array(r['rand']), np.array(r['half'], bool)
        if len(s) < 20:
            continue
        e = (s - q) * 100 * (1 if side == 'buy' else -1)       # + 면 신호에 정보가 있음 (파는 신호는 더 내려야 +)
        out.setdefault(side, {}).setdefault(name, {})[hname] = {
            'n': int(len(s)), 'sig': float(s.mean() * 100), 'rand': float(q.mean() * 100), 'edge': float(e.mean()),
            'f': float(e[~half].mean()) if (~half).any() else None, 'b': float(e[half].mean()) if half.any() else None,
            'win': float((s > 0).mean() * 100)}
    return out


def show(title, res):
    print(f"\n######## {title}")
    for side, label in (('buy', '사는 신호 (+ 면 아무 봉보다 나음)'), ('sell', '파는 신호 (+ 면 신호 뒤 실제로 더 내림)')):
        print(f"  ── {label}")
        for name, hs in res.get(side, {}).items():
            cells = []
            for hname in ('4시간', '1일', '3일'):
                x = hs.get(hname)
                if x:
                    both = '◎' if x['f'] is not None and x['b'] is not None and x['f'] > 0 and x['b'] > 0 else ('×' if x['f'] is not None and x['b'] is not None and x['f'] < 0 and x['b'] < 0 else ' ')
                    cells.append(f"{hname} {x['edge']:+5.2f}{both}")
            n = next(iter(hs.values()))['n']
            print(f"    {name:<30} {n:6d}번 | " + ' · '.join(cells))


def main():
    m15 = load_15m()
    print(f"15분봉 {len(m15)}종")
    out = {}
    out['15m_1y'] = evaluate(m15, 4)
    show('15분봉 · 1년 · 74종', out['15m_1y'])
    h1 = {k: to_hour(t, a, 3600) for k, (t, a) in m15.items()}
    out['1h_1y'] = evaluate(h1, 1)
    show('1시간봉 · 1년 · 74종 (15분봉을 묶음)', out['1h_1y'])
    h1b = load_1m_hour()
    print(f"\n1분봉 → 1시간봉 {len(h1b)}종")
    out['1h_70d'] = evaluate(h1b, 1)
    show('1시간봉 · 약 70일 · 288종 (1분봉을 묶음)', out['1h_70d'])
    (HERE / 'outputs' / 'probe32_intraday_ta.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
