"""
probe20: 블로그 11편 규칙(spike-prone-v1) 숫자 손보기 — 앞 60일로 고르고 뒤 30일로 확인

probe18 1분봉 캐시(281종 · 90일). StraHub 엔진(utils/spike_rule_backtest.py)과 같은 사고팔기를 numpy로 빠르게:
  사기: '자주 튀던 코인 특성' 그리고 '급등 직전 모습'에 새로 맞은 다음 분 시작가 (+불리한 체결 0.2%, 수수료 0.05%)
  팔기: 손절 / 익절 (같은 분에 둘 다면 손절), 가진 동안은 새로 사지 않음
  (끝까지 안 팔린 거래는 마지막 가격으로 셈 · 급등 세트는 뺌 — 기본 규칙에서 90일 동안 11번뿐이라 결과에 거의 영향 없음)
비교: 같은 코인·같은 기간에서 같은 횟수만큼 아무 때나 산 경우(같은 손절·익절)
결과: outputs/probe20_rule_tune.json
"""
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'AutoTrading_Flask'))

import probe18_spike_traits as p18  # noqa: E402
from probe19_rule_backtest import facts_for  # noqa: E402

OUT = HERE / 'outputs' / 'probe20_rule_tune.json'
FEE, SLIP, DAY = 0.0005, 0.002, 1440
SPLIT_DAYS = 60
SPLIT = SPLIT_DAYS * DAY                     # 이 분 번호 전에 산 거래 = 고르기용, 뒤 = 확인용
RISES = (3, 5, 8)
MULTS = (5, 10, 20)
STOPS = (3, 5, 8, 12, 20)
TAKES = (5, 10, 20, 40, None)                 # None = 익절 없음 (손절이나 기간 끝까지)
PRONE = {'max_value': 100, 'new_days': 90, 'max_price': 10}
RNG = np.random.default_rng(20)


def coin_arrays(market):
    b = p18.load(market)
    if b is None:
        return None
    f = facts_for(market)
    v = b['v']
    cs = np.concatenate([[0.0], np.cumsum(v)])
    idx = np.arange(p18.N)
    lo = np.maximum(0, idx + 1 - DAY)
    value24 = cs[idx + 1] - cs[lo]
    v15 = cs[idx + 1] - cs[np.maximum(0, idx + 1 - 15)]
    c = b['c']
    ago = np.full(p18.N, np.nan)
    ago[60:] = c[:-60]
    t = p18.START + idx * 60
    if f['listed_old']:
        new = np.zeros(p18.N, bool)
    elif f['listed_ts']:
        new = (t - f['listed_ts']) // 86400 < PRONE['new_days']
    else:
        new = np.zeros(p18.N, bool)
    trait = new | (c < PRONE['max_price']) | (f['on_binance'] is False)
    with np.errstate(invalid='ignore', divide='ignore'):
        prone = (value24 > 0) & (value24 <= PRONE['max_value'] * 1e8) & trait
        rise = (c / ago - 1) * 100
        mult = (v15 / 15) / (value24 / DAY)
    ready = idx >= b['first'] + DAY - 1
    return {'b': b, 'prone': prone & ready, 'rise': np.nan_to_num(rise, nan=-1e9),
            'mult': np.nan_to_num(mult, nan=0), 'start': b['first'] + DAY - 1, 'ready': ready}


def transitions(match, start):
    """새로 맞은 분 (시작 때 이미 맞던 것은 빼기)"""
    m = match.copy()
    prev = np.concatenate([[True], m[:-1]])
    prev[start] = True
    out = np.flatnonzero(m & ~prev)
    return out[(out >= start) & (out + 1 < p18.N)]


def exit_of(b, e, sl, tp):
    """e 분에 신호 → e+1 시작가에 삼. (판 분, 수익률 %, 어떻게)"""
    o, h, l, c = b['o'], b['h'], b['l'], b['c']
    P = o[e + 1] * (1 + SLIP)
    s_lv = P * (1 - sl / 100)
    t_lv = P * (1 + tp / 100) if tp else np.inf
    k0 = e + 1
    step = 2048
    while k0 < p18.N:
        k1 = min(p18.N, k0 + step)
        hit_s = l[k0:k1] <= s_lv
        hit_t = h[k0:k1] >= t_lv
        any_ = hit_s | hit_t
        if any_.any():
            j = int(np.argmax(any_))
            k = k0 + j
            if hit_s[j]:
                px = min(s_lv, o[k]) * (1 - SLIP)
                how = 'stop'
            else:
                px = max(t_lv, o[k]) * (1 - SLIP)
                how = 'take'
            return k, (px * (1 - FEE) / (P / (1 - FEE)) - 1) * 100, how
        k0 = k1
        step *= 4
    px = c[-1]
    return p18.N - 1, (px * (1 - FEE) / (P / (1 - FEE)) - 1) * 100, 'open'


def run_entries(b, sigs, sl, tp):
    """신호들을 차례로: 가진 동안 온 신호는 건너뜀 → [(산 분, 수익률, how)]"""
    out, free = [], -1
    for e in sigs:
        if e <= free:
            continue
        k, r, how = exit_of(b, int(e), sl, tp)
        out.append((int(e), r, how))
        free = k
    return out


def stats(trips):
    """끝까지 안 팔린 거래도 마지막 가격으로 셈 (빼면 넓은 손절이 좋아 보이는 치우침이 생김)"""
    g = [r for _, r, how in trips]
    if not g:
        return {'n': 0, 'mean': None, 'win': None, 'sum': 0.0, 'open': 0}
    a = np.array(g)
    return {'n': len(g), 'mean': float(a.mean()), 'win': float((a > 0).mean() * 100), 'median': float(np.median(a)),
            'sum': float(a.sum()), 'open': sum(how == 'open' for *_, how in trips)}


def main():
    markets = sorted(x.stem for x in p18.DATA.glob('KRW-*.npz') if x.stem not in p18.PEGGED)
    coins = {}
    for m in markets:
        a = coin_arrays(m)
        if a is not None and a['start'] + 120 < p18.N:
            coins[m] = a
    print('coins', len(coins), flush=True)

    rows = []
    for rise, mult in product(RISES, MULTS):
        sigs = {m: transitions(a['prone'] & (a['rise'] >= rise) & (a['mult'] >= mult), a['start']) for m, a in coins.items()}
        for sl, tp in product(STOPS, TAKES):
            tr, te = [], []
            for m, a in coins.items():
                for e, r, how in run_entries(a['b'], sigs[m], sl, tp):
                    (tr if e < SPLIT else te).append((e, r, how))
            rows.append({'rise': rise, 'mult': mult, 'sl': sl, 'tp': tp, 'train': stats(tr), 'test': stats(te)})
        print(f'rise {rise} mult {mult} done', flush=True)

    ok = [r for r in rows if r['train']['n'] >= 100]
    ok.sort(key=lambda r: -r['train']['mean'])
    best = ok[0]
    default = next(r for r in rows if (r['rise'], r['mult'], r['sl'], r['tp']) == (3, 5, 5, 10))

    # 아무 때나 산 경우: 고른 규칙과 같은 코인·같은 기간 쪽(뒤 30일)에서 같은 횟수 · 같은 손절·익절
    def random_baseline(cfg, part, reps=5):
        means = []
        for _ in range(reps):
            res = []
            for m, a in coins.items():
                sigs = transitions(a['prone'] & (a['rise'] >= cfg['rise']) & (a['mult'] >= cfg['mult']), a['start'])
                sigs = sigs[(sigs >= SPLIT) if part == 'test' else (sigs < SPLIT)]
                n = len(run_entries(a['b'], sigs, cfg['sl'], cfg['tp']))
                if not n:
                    continue
                lo, hi = (max(a['start'], SPLIT), p18.N - 2) if part == 'test' else (a['start'], SPLIT - 1)
                if hi <= lo:
                    continue
                pick = np.sort(RNG.integers(lo, hi, size=n * 3))
                res += run_entries(a['b'], pick, cfg['sl'], cfg['tp'])[:n]
            means.append(stats(res)['mean'])
        return float(np.mean([x for x in means if x is not None]))

    out = {
        'meta': {'split_days': SPLIT_DAYS, 'window': [p18.START, p18.END], 'coins': len(coins), 'prone': PRONE,
                 'grid': {'rise': RISES, 'mult': MULTS, 'sl': STOPS, 'tp': TAKES}},
        'default': default, 'best': best, 'top10': ok[:10],
        'test_positive_share': sum(r['test']['mean'] is not None and r['test']['mean'] > 0 for r in rows) / len(rows) * 100,
        'train_top10_test_positive': sum(r['test']['mean'] is not None and r['test']['mean'] > 0 for r in ok[:10]),
        'random_default_test': random_baseline(default, 'test'),
        'random_best_test': random_baseline(best, 'test'),
        'random_best_train': random_baseline(best, 'train'),
        'rows': rows,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
