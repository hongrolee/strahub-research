"""
probe25: 새 봇3(업비트) 설정 검증 — '급등이 잦던 코인' 매수 + 손절 · 익절
- 자료: StraBot 신뢰도 기록용 일봉 (signal_log.db candles, 원화 코인 약 250종 · 약 400일)
- 7일마다 시작: 그날 '급등이 잦던 코인'(거래대금 ≤100억 & (상장 90일 미만 | 10원 미만 | 바이낸스 없음))을 같은 금액씩 산다
  (비교: 같은 날 모든 코인)
- 들고 있는 동안 손절 −S% / 익절 +T% 에 닿으면 그 값에 판다 (일봉 저가 · 고가로 먼저 닿았는지, 같은 날 둘 다면 손절 먼저 = 보수적)
- 비용 한쪽 0.15% · 30일 · 90일
- 한계: 봇3의 '급등 때 30% 팔고 싸지면 다시 사기'(세트)는 1분봉이 필요해 여기선 빼고 따로 적는다 (probe21: 세트 보탬 1~3%)
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DB = Path(r'E:\MAYAI\AutoTrading_Flask\trading_bot_template\dev_preview\data\signal_log.db')
COST = 0.0015
STOPS = [None, 10, 15, 20, 30, 50]
TAKES = [None, 20, 30, 50, 100, 200]


def load():
    con = sqlite3.connect(DB)
    rows = con.execute('SELECT market, day, h, l, c, v FROM candles ORDER BY market, day').fetchall()
    days = sorted({r[1] for r in rows})
    di = {d: i for i, d in enumerate(days)}
    mk = sorted({r[0] for r in rows})
    mi = {m: i for i, m in enumerate(mk)}
    H = np.full((len(mk), len(days)), np.nan)
    L, C, V = H.copy(), H.copy(), H.copy()
    for m, d, h, l, c, v in rows:
        i, j = mi[m], di[d]
        H[i, j], L[i, j], C[i, j], V[i, j] = h, l, c, v or 0
    first = np.array([np.argmax(~np.isnan(C[i])) for i in range(len(mk))])
    try:
        binance = set(json.loads((HERE / 'data' / 'binance_bases.json').read_text(encoding='utf-8')))
    except Exception:
        binance = set()
    return mk, days, H, L, C, V, first, binance


def spike_prone(i, t, mk, C, V, first, binance, start0):
    """그날 기준 '급등이 잦던 코인' (봇 기본 숫자: 거래대금 100억 이하, 상장 90일 미만 · 10원 미만 · 바이낸스 없음 중 하나)"""
    if np.isnan(C[i, t]) or V[i, t] > 100e8:
        return False
    cur = mk[i].split('-', 1)[1]
    new = first[i] > start0 and t - first[i] < 90          # 자료 맨 앞에 이미 있던 코인은 '오래된' 것으로 봄
    return new or C[i, t] < 10 or cur not in binance


def run(mk, days, H, L, C, V, first, binance, hold, pick):
    res = {(s, t): [] for s in STOPS for t in TAKES}
    hits = {(s, t): [0, 0] for s in STOPS for t in TAKES}
    nstart = 0
    start0 = 0
    for t0 in range(30, len(days) - hold, 7):
        idx = [i for i in range(len(mk)) if not np.isnan(C[i, t0]) and t0 - first[i] >= 5
               and (pick == 'all' or spike_prone(i, t0, mk, C, V, first, binance, start0))]
        if not idx:
            continue
        nstart += 1
        for i in idx:
            e = C[i, t0]
            ll, hh = L[i, t0 + 1:t0 + hold + 1], H[i, t0 + 1:t0 + hold + 1]
            cc = C[i, t0 + hold]
            if np.isnan(cc):                                  # 중간에 사라진 코인(상장폐지): 마지막 값
                ok = np.flatnonzero(~np.isnan(C[i, t0:t0 + hold + 1]))
                cc = C[i, t0 + ok[-1]] if len(ok) else e
            for s in STOPS:
                ks = np.flatnonzero(ll <= e * (1 - s / 100)) if s else []
                k_s = ks[0] if len(ks) else 10 ** 9
                for tk in TAKES:
                    kt = np.flatnonzero(hh >= e * (1 + tk / 100)) if tk else []
                    k_t = kt[0] if len(kt) else 10 ** 9
                    if k_s <= k_t and k_s < 10 ** 9:
                        px = e * (1 - s / 100); hits[(s, tk)][0] += 1
                    elif k_t < 10 ** 9:
                        px = e * (1 + tk / 100); hits[(s, tk)][1] += 1
                    else:
                        px = cc
                    res[(s, tk)].append((px * (1 - COST)) / (e * (1 + COST)) - 1)
    out = []
    for k, v in res.items():
        a = np.array(v)
        if not len(a):
            continue
        out.append({'stop': k[0], 'take': k[1], 'n': len(a), 'avg': float(a.mean() * 100), 'median': float(np.median(a) * 100),
                    'p10': float(np.percentile(a, 10) * 100), 'win': float((a > 0).mean() * 100),
                    'stop_hit': hits[k][0] / len(a) * 100, 'take_hit': hits[k][1] / len(a) * 100})
    return out, nstart


def show(title, rows, nstart):
    print(f'\n== {title} (시작 {nstart}번)')
    base = next(r for r in rows if r['stop'] is None and r['take'] is None)
    print(f"그냥 들고 있기: 평균 {base['avg']:+.1f}% · 중앙 {base['median']:+.1f}% · 하위10% {base['p10']:+.1f}% · 이김 {base['win']:.0f}% (n={base['n']})")
    best = sorted(rows, key=lambda r: -r['avg'])[:5]
    for r in sorted(rows, key=lambda r: (r['stop'] or 999, r['take'] or 999)):
        mark = ' ★' if r in best else ''
        cur = ' ← 지금 설정' if r['stop'] == 20 and r['take'] == 50 else ''
        print(f"  손절 {str(r['stop'] or '-'):>3} 익절 {str(r['take'] or '-'):>4} | 평균 {r['avg']:+6.1f}% 중앙 {r['median']:+6.1f}% 하위10% {r['p10']:+6.1f}% "
              f"이김 {r['win']:3.0f}% | 손절 {r['stop_hit']:4.0f}% 익절 {r['take_hit']:4.0f}%{mark}{cur}")


def main():
    data = load()
    mk, days = data[0], data[1]
    print(f'코인 {len(mk)} · {days[0]} ~ {days[-1]} ({len(days)}일) · 바이낸스 목록 {len(data[7])}개')
    out = {}
    for hold in (30, 90):
        for pick in ('spike', 'all'):
            rows, n = run(*data, hold, pick)
            show(f"{'급등이 잦던 코인' if pick == 'spike' else '모든 코인'} · {hold}일", rows, n)
            out[f'{pick}_{hold}'] = rows
    (HERE / 'outputs' / 'probe25_bot3_settings.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
