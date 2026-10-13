"""
probe37: StraBot 조건 66개를 '매수 조건'으로 같은 잣대에 — 시뮬레이션 봇으로 올릴 후보 고르기 (1차 거르기)
- 자료: StraBot 신뢰도 기록 signal_log.db (읽기 전용) — 기본 숫자 그대로 날마다 '새로 뜬' 신호 (2026-01-03 ~), 일봉 candles
- 사기: 신호 뜬 날 종가 / 팔기 두 가지: ① 10일 들고 있기  ② 손절 15 · 익절 50 · 최대 30일 (새 봇 3 방식, 일봉 저가 · 고가로 먼저 닿았는지)
- 비교: ① 10일 = 같은 날 사서 같은 날 판 '모든 코인 평균' ② 봇식 = 같은 날 아무 코인 5개를 똑같은 손절 · 익절로 (비용은 10일 쪽만 0.3%, 봇식은 양쪽 같아 뺌)
- 앞 절반 · 뒤 절반(신호 날짜로 나눔) 따로 · 신호가 몰린 날 수도 셈 (같은 날 수십 개면 사실상 한 번)
결과: outputs/probe37_condition_screen.json
"""
import json
import sqlite3
from pathlib import Path

import numpy as np

from probe25_bot3_settings import load, DB

HERE = Path(__file__).resolve().parent
COST = 0.003


def main():
    mk, days, H, L, C, V, first, binance = load()
    mi = {m: i for i, m in enumerate(mk)}
    di = {d: i for i, d in enumerate(days)}
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    rows = con.execute('SELECT cond, market, day, dir FROM signals').fetchall()
    T = len(days)
    fwd10 = np.full((len(mk), T), np.nan)
    fwd10[:, :T - 10] = C[:, 10:] / C[:, :-10] - 1
    with np.errstate(invalid='ignore'):
        mkt10 = np.nanmean(fwd10, axis=0)

    def bot_exit(i, t0):
        """손절 15 · 익절 50 · 최대 30일 → (수익, 판 날)"""
        e = C[i, t0]
        for k in range(1, 31):
            t = t0 + k
            if t >= T or np.isnan(C[i, t]):
                return (C[i, t - 1] / e - 1) if not np.isnan(C[i, t - 1]) else 0.0, t - 1
            if L[i, t] <= e * 0.85:
                return -0.15, t
            if H[i, t] >= e * 1.5:
                return 0.5, t
        return C[i, t0 + 30] / e - 1, t0 + 30

    RNG = np.random.default_rng(37)
    alive = [np.flatnonzero(~np.isnan(C[:, t]) & (t - first >= 5)) for t in range(T)]
    by = {}
    dirs = {}
    for cond, m, d, dr in rows:
        if m in mi and d in di:
            by.setdefault(cond, []).append((mi[m], di[d]))
            dirs[cond] = dr
    all_days = sorted({t for v in by.values() for _, t in v})
    mid = all_days[len(all_days) // 2]
    out = []
    for cond, sig in by.items():
        e10, eb, half, ndays, r10, rb = [], [], [], set(), [], []
        for i, t in sig:
            if t + 10 >= T or np.isnan(C[i, t]) or np.isnan(fwd10[i, t]):
                continue
            a = fwd10[i, t] - COST
            r10.append(a)
            e10.append(a - (mkt10[t] - 0))
            rb_, te = bot_exit(i, t)
            # 대조: 같은 날 아무 코인 5개를 사서 똑같이 손절 15 · 익절 50 · 최대 30일로 판 평균 (손절이 손실을 잘라 주는 몫은 양쪽에 똑같이)
            pool = alive[t]
            pick = RNG.choice(pool, size=min(5, len(pool)), replace=False) if len(pool) else []
            ctrl = [bot_exit(j, t)[0] for j in pick if j != i]
            mk_b = float(np.mean(ctrl)) if ctrl else 0.0
            rb.append(rb_ - COST)
            eb.append(rb_ - mk_b)
            half.append(t >= mid)
            ndays.add(t)
        if len(e10) < 20:
            continue
        e10, eb, half = np.array(e10) * 100, np.array(eb) * 100, np.array(half)
        out.append({'cond': cond, 'dir': dirs[cond], 'n': len(e10), 'days': len(ndays),
                    'ret10': float(np.mean(r10) * 100), 'edge10': float(e10.mean()), 'edge10_f': float(e10[~half].mean()) if (~half).any() else None,
                    'edge10_b': float(e10[half].mean()) if half.any() else None,
                    'retb': float(np.mean(rb) * 100), 'edgeb': float(eb.mean()), 'edgeb_f': float(eb[~half].mean()) if (~half).any() else None,
                    'edgeb_b': float(eb[half].mean()) if half.any() else None})
    out.sort(key=lambda r: -r['edgeb'])
    print(f"신호 기록 {all_days and days[all_days[0]]} ~ {days[all_days[-1]]}, 앞/뒤 나눔 {days[mid]} · 조건 {len(out)}개 (비용 왕복 0.3% 뺌, '몫' = 같은 날 모든 코인 평균보다)")
    print(f"{'조건':<22}{'쪽':>5}{'신호':>7}{'날':>5} | {'10일 몫':>8}{'앞':>7}{'뒤':>7} | {'봇식 몫':>8}{'앞':>7}{'뒤':>7} | 둘 다")
    for r in out:
        both = all(x is not None and x > 0 for x in (r['edge10_f'], r['edge10_b'], r['edgeb_f'], r['edgeb_b']))
        r['robust'] = both
        f = lambda x: f"{x:+7.2f}" if x is not None else '    -  '   # noqa: E731
        print(f"{r['cond']:<22}{r['dir']:>5}{r['n']:>7}{r['days']:>5} | {r['edge10']:+8.2f}{f(r['edge10_f'])}{f(r['edge10_b'])} | {r['edgeb']:+8.2f}{f(r['edgeb_f'])}{f(r['edgeb_b'])} | {'◎' if both else ''}")
    (HERE / 'outputs' / 'probe37_condition_screen.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
