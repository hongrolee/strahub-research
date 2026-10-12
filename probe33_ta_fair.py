"""
probe33: probe30 · 31 · 32 의 '사는 신호' 대조를 고침 — 공정한 비교 = 같은 시각 다른 코인들의 평균 수익
- 전에는 대조를 '같은 코인 앞뒤 N일 안 아무 날'로 잡았는데, 신호 앞 시간도 섞여서 신호가 이미 올라 버린 몫을 대조가 먹음
  → 오른 뒤 뜨는 신호(돌파 · 추세)는 정보가 없어도 나빠 보이고, 내린 뒤 뜨는 신호는 좋아 보이는 치우침
- 이번: 신호 봉 종가에 산 뒤 H 봉 수익 − 같은 봉 종가에 산 모든 코인 평균 수익 (시장 흐름만 빼고, 앞을 보지 않음)
  사는 신호는 + 면 정보 있음 / 파는 신호는 신호 뒤 덜 오르면(− 면) 정보 있음 → 부호를 바꿔 + 가 좋게
- 자료: 일봉 1년 284종(signal_log.db) · 15분봉 1년 74종 · 그 1시간봉 · 1분봉 70일 288종의 1시간봉
결과: outputs/probe33_ta_fair.json
"""
import json
from pathlib import Path

import numpy as np

from probe25_bot3_settings import load
from probe32_intraday_ta import load_15m, load_1m_hour, to_hour, signals

HERE = Path(__file__).resolve().parent
COST = 0.0015


def panel(series):
    """{coin: (t, a)} → 시각 격자에 맞춘 종가 행렬 · 코인별 그 격자 위치"""
    grid = np.unique(np.concatenate([t for t, _ in series.values()]))
    gi = {int(x): k for k, x in enumerate(grid)}
    coins = list(series)
    C = np.full((len(coins), len(grid)), np.nan)
    pos = {}
    for j, cn in enumerate(coins):
        t, a = series[cn]
        p = np.array([gi[int(x)] for x in t])
        C[j, p] = a[:, 3]
        pos[cn] = p
    return coins, grid, C, pos


def evaluate(series, holds):
    coins, grid, C, pos = panel(series)
    fwd = {h: C[:, h:] / C[:, :-h] - 1 for h in holds.values()}
    mkt = {h: np.nanmean(f, axis=0) for h, f in fwd.items()}
    mid = (grid[0] + grid[-1]) // 2
    acc = {}
    for j, cn in enumerate(coins):
        t, a = series[cn]
        if len(t) < 300:
            continue
        buy, sell = signals(a)
        p = pos[cn]
        for side, group in (('buy', buy), ('sell', sell)):
            for name, S in group.items():
                idx = np.flatnonzero(S)
                idx = idx[idx >= 60]
                for hname, H in holds.items():
                    g = p[idx]
                    g = g[g < len(grid) - H]
                    f = fwd[H][j, g] - 2 * COST
                    m = mkt[H][g] - 2 * COST
                    ok = ~np.isnan(f) & ~np.isnan(m)
                    rec = acc.setdefault((side, name, hname), {'e': [], 's': [], 'half': []})
                    e = (f[ok] - m[ok]) * 100 * (1 if side == 'buy' else -1)
                    rec['e'].extend(e.tolist()); rec['s'].extend((f[ok] * 100).tolist()); rec['half'].extend((grid[g[ok]] >= mid).tolist())
    out = {}
    for (side, name, hname), r in acc.items():
        e, s, half = np.array(r['e']), np.array(r['s']), np.array(r['half'], bool)
        if len(e) < 20:
            continue
        out.setdefault(side, {}).setdefault(name, {})[hname] = {
            'n': int(len(e)), 'edge': float(e.mean()), 'sig': float(s.mean()),
            'f': float(e[~half].mean()) if (~half).any() else None, 'b': float(e[half].mean()) if half.any() else None}
    return out


def show(title, res, order):
    print(f"\n######## {title}")
    for side, label in (('buy', '사는 신호 (+ 면 같은 때 다른 코인보다 더 오름)'), ('sell', '파는 신호 (+ 면 같은 때 다른 코인보다 덜 오름)')):
        print(f"  ── {label}")
        for name, hs in res.get(side, {}).items():
            cells = []
            for hname in order:
                x = hs.get(hname)
                if x:
                    mark = '◎' if x['f'] is not None and x['b'] is not None and x['f'] > 0 and x['b'] > 0 else ('×' if x['f'] is not None and x['b'] is not None and x['f'] < 0 and x['b'] < 0 else ' ')
                    cells.append(f"{hname} {x['edge']:+5.2f}{mark}")
            n = next(iter(hs.values()))['n']
            print(f"    {name:<30} {n:6d}번 | " + ' · '.join(cells))


def daily_series():
    mk, days, H, L, C, V, first, binance = load()
    t = np.arange(len(days)) * 86400
    out = {}
    for i, m in enumerate(mk):
        ok = ~np.isnan(C[i])
        if ok.sum() < 120:
            continue
        a = np.stack([C[i], H[i], L[i], C[i], V[i]], 1)[ok]
        out[m] = (t[ok], a)
    return out


def main():
    out = {}
    d = daily_series()
    out['1d'] = evaluate(d, {'1일': 1, '5일': 5, '30일': 30})
    show(f'일봉 · 1년 · {len(d)}종', out['1d'], ('1일', '5일', '30일'))
    m15 = load_15m()
    hh = {'4시간': 16, '1일': 96, '3일': 288}
    out['15m'] = evaluate(m15, hh)
    show('15분봉 · 1년 · 74종', out['15m'], tuple(hh))
    h1 = {k: to_hour(t, a, 3600) for k, (t, a) in m15.items()}
    hh1 = {'4시간': 4, '1일': 24, '3일': 72}
    out['1h'] = evaluate(h1, hh1)
    show('1시간봉 · 1년 · 74종', out['1h'], tuple(hh1))
    h1b = load_1m_hour()
    out['1h_70d'] = evaluate(h1b, hh1)
    show(f'1시간봉 · 약 70일 · {len(h1b)}종', out['1h_70d'], tuple(hh1))
    (HERE / 'outputs' / 'probe33_ta_fair.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
