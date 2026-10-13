"""
probe38: 조건을 겹치면(둘 다 맞을 때만 사기) 나아질까? — probe37 과 같은 자료 · 같은 비교
- 조합: 신호 50번 넘는 조건들 중에서 둘씩, 같은 코인에서 3일 안에 둘 다 뜬 날(늦게 뜬 날) 사기
- 팔기 · 비교: probe37 ② 봇식 (손절 15 · 익절 50 · 최대 30일, 같은 날 아무 코인 5개를 같은 방식으로) + ① 10일 (같은 날 모든 코인 평균, 비용 0.3%)
- 우연 기준: 같은 수의 '가짜 조합' — 조건 둘의 신호를 무작위로 섞어 짝지은 것(같은 신호 수)으로 몇 개가 좋아 보이는지
결과: outputs/probe38_condition_combos.json
"""
import json
import sqlite3
from itertools import combinations
from pathlib import Path

import numpy as np

from probe25_bot3_settings import load, DB

HERE = Path(__file__).resolve().parent
COST = 0.003
RNG = np.random.default_rng(38)


def main():
    mk, days, H, L, C, V, first, binance = load()
    mi = {m: i for i, m in enumerate(mk)}
    di = {d: i for i, d in enumerate(days)}
    T = len(days)
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    by = {}
    for cond, m, d, dr in con.execute('SELECT cond, market, day, dir FROM signals'):
        if m in mi and d in di:
            by.setdefault(cond, set()).add((mi[m], di[d]))
    conds = sorted(c for c, s in by.items() if len(s) >= 50 and not c.startswith('my_'))
    fwd10 = np.full((len(mk), T), np.nan)
    fwd10[:, :T - 10] = C[:, 10:] / C[:, :-10] - 1
    with np.errstate(invalid='ignore'):
        mkt10 = np.nanmean(fwd10, axis=0)
    alive = [np.flatnonzero(~np.isnan(C[:, t]) & (t - first >= 5)) for t in range(T)]
    cache = {}

    def bot_exit(i, t0):
        key = (i, t0)
        if key in cache:
            return cache[key]
        e = C[i, t0]
        r = None
        for k in range(1, 31):
            t = t0 + k
            if t >= T or np.isnan(C[i, t]):
                r = (C[i, t - 1] / e - 1) if not np.isnan(C[i, t - 1]) else 0.0
                break
            if L[i, t] <= e * 0.85:
                r = -0.15
                break
            if H[i, t] >= e * 1.5:
                r = 0.5
                break
        if r is None:
            r = C[i, t0 + 30] / e - 1
        cache[key] = r
        return r

    all_days = sorted({t for s in by.values() for _, t in s})
    mid = all_days[len(all_days) // 2]

    def score(sig):
        e10, eb, half = [], [], []
        for i, t in sig:
            if t + 30 >= T or np.isnan(C[i, t]) or np.isnan(fwd10[i, t]):
                continue
            e10.append(fwd10[i, t] - COST - mkt10[t])
            pool = alive[t]
            pick = RNG.choice(pool, size=min(5, len(pool)), replace=False)
            ctrl = [bot_exit(j, t) for j in pick if j != i]
            eb.append(bot_exit(i, t) - (np.mean(ctrl) if ctrl else 0.0))
            half.append(t >= mid)
        if len(e10) < 20:
            return None
        e10, eb, half = np.array(e10) * 100, np.array(eb) * 100, np.array(half)
        if half.all() or (~half).all():
            return None
        return {'n': len(e10), 'e10': float(e10.mean()), 'eb': float(eb.mean()),
                'ok': bool(e10[~half].mean() > 0 and e10[half].mean() > 0 and eb[~half].mean() > 0 and eb[half].mean() > 0)}

    def pair(a, b):
        """같은 코인에서 3일 안에 둘 다 뜬 날 (늦게 뜬 날)"""
        bi = {}
        for i, t in by[b]:
            bi.setdefault(i, []).append(t)
        out = set()
        for i, t in by[a]:
            for u in bi.get(i, []):
                if abs(u - t) <= 3:
                    out.add((i, max(t, u)))
        return out

    singles = {c: score(by[c]) for c in conds}
    combos, fakes = [], []
    for a, b in combinations(conds, 2):
        s = pair(a, b)
        if len(s) < 20:
            continue
        r = score(s)
        if not r:
            continue
        combos.append({'a': a, 'b': b, **r})
        # 가짜 조합: 같은 개수의 신호를 a ∪ b 신호에서 아무렇게나 뽑음 (겹친다는 정보만 뺌)
        pool = list(by[a] | by[b])
        idx = RNG.choice(len(pool), size=min(len(s), len(pool)), replace=False)
        f = score([pool[k] for k in idx])
        if f:
            fakes.append(f)
    ok_single = sum(1 for v in singles.values() if v and v['ok'])
    ok_combo = sum(1 for r in combos if r['ok'])
    ok_fake = sum(1 for r in fakes if r['ok'])
    print(f"조건 {len(conds)}개 · 하나씩: 앞뒤 · 두 방식 모두 + 인 것 {ok_single}개 ({ok_single / len(conds) * 100:.0f}%)")
    print(f"둘씩 겹친 조합 {len(combos)}개: 모두 + 인 것 {ok_combo}개 ({ok_combo / max(1, len(combos)) * 100:.0f}%)")
    print(f"가짜 조합(겹침 정보 없이 같은 수) {len(fakes)}개: 모두 + 인 것 {ok_fake}개 ({ok_fake / max(1, len(fakes)) * 100:.0f}%)")
    print(f"평균 봇식 몫: 하나씩 {np.mean([v['eb'] for v in singles.values() if v]):+.2f}%p · 겹친 조합 {np.mean([r['eb'] for r in combos]):+.2f}%p · 가짜 조합 {np.mean([r['eb'] for r in fakes]):+.2f}%p")
    print(f"평균 신호 수: 하나씩 {np.mean([v['n'] for v in singles.values() if v]):.0f} · 겹친 조합 {np.mean([r['n'] for r in combos]):.0f}")
    # 하나씩 좋았던 5개끼리 겹치면?
    good = ['desc_scallop', 'hs_top', 'double_top', 'streak_down', 'ma_cross_down']
    print("\n통과한 5개끼리 겹친 조합 (봇식 몫 · 10일 몫 · 신호 수 · 모두 +):")
    for r in combos:
        if r['a'] in good and r['b'] in good:
            print(f"  {r['a']} + {r['b']}: {r['eb']:+.2f} · {r['e10']:+.2f} · {r['n']} · {'◎' if r['ok'] else ''}")
    print("\n겹친 조합 중 모두 + 이고 봇식 몫 큰 순 10개:")
    for r in sorted([r for r in combos if r['ok']], key=lambda r: -r['eb'])[:10]:
        print(f"  {r['a']} + {r['b']}: 봇식 {r['eb']:+.2f} · 10일 {r['e10']:+.2f} · 신호 {r['n']} "
              f"(하나씩: {singles[r['a']]['eb']:+.2f} / {singles[r['b']]['eb']:+.2f})")
    (HERE / 'outputs' / 'probe38_condition_combos.json').write_text(json.dumps({'singles': singles, 'combos': combos}, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
