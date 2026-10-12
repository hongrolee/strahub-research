"""
probe31: 트레이딩뷰 인기 지표 두 개 검증 — 일봉 (signal_log.db, probe25 자료)
 A) Squeeze Momentum [LazyBear] — BB(20, 2) 가 KC(20, 1.5 · 진폭=TR) 안에 들어가면 스퀴즈, 빠져나오면 '터짐'
    모멘텀 = linreg(종가 − ((20봉 최고+최저)/2 + 20봉 평균)/2, 20)
    사는 신호: ① 스퀴즈 터짐 & 모멘텀 > 0  ② 터짐 & 모멘텀 > 0 & 오르는 중  ③ 모멘텀이 0 위로
    파는 신호: ① 모멘텀이 0 아래로  ② 터짐 & 모멘텀 < 0
 B) Smart Money Concepts [LuxAlgo] — 구조(BOS · CHoCH): 고점/저점(내부 5봉 · 큰 흐름 50봉)을 종가가 넘으면
    추세와 같은 쪽 = BOS, 반대 쪽 = CHoCH
    사는 신호: 위로 CHoCH · 위로 BOS (내부 · 큰 흐름) / 파는 신호: 아래로 CHoCH · 아래로 BOS
검증 (probe30 과 같게): 사는 신호 → 30일(손절 15 · 익절 50) vs 같은 코인 앞뒤 15일 아무 날
 파는 신호 → probe25 방식으로 산 코인을 신호 날 종가에 팖 vs 같은 비율로 아무 날 (비율을 맞춰 가며)
비용 한쪽 0.15%. 결과: outputs/probe31_squeeze_smc.json
"""
import json
from pathlib import Path

import numpy as np

from probe25_bot3_settings import load, spike_prone, COST

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(31)
STOP, TAKE = 15, 50


# ── 지표 ──
def rolling(a, n, fn):
    out = np.full_like(a, np.nan)
    for t in range(n - 1, len(a)):
        w = a[t - n + 1:t + 1]
        if not np.isnan(w).any():
            out[t] = fn(w)
    return out


def linreg_last(a, n):
    """트레이딩뷰 linreg(src, n, 0): 최근 n개에 맞춘 직선의 마지막 값"""
    x = np.arange(n)
    xm = x.mean()
    den = ((x - xm) ** 2).sum()
    out = np.full_like(a, np.nan)
    for t in range(n - 1, len(a)):
        w = a[t - n + 1:t + 1]
        if np.isnan(w).any():
            continue
        b = ((x - xm) * (w - w.mean())).sum() / den
        out[t] = w.mean() + b * (n - 1 - xm)
    return out


def squeeze(h, l, c, n=20, bb=2.0, kc=1.5):
    basis = rolling(c, n, np.mean)
    dev = bb * rolling(c, n, np.std)
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.fmax(h - l, np.fmax(np.abs(h - pc), np.abs(l - pc)))
    rma = rolling(tr, n, np.mean)
    on = (basis - dev > basis - rma * kc) & (basis + dev < basis + rma * kc)
    off = (basis - dev < basis - rma * kc) & (basis + dev > basis + rma * kc)
    hh, ll = rolling(h, n, np.max), rolling(l, n, np.min)
    val = linreg_last(c - ((hh + ll) / 2 + basis) / 2, n)
    return on, off, val


def structure(h, l, c, size):
    """LuxAlgo SMC 구조: (위 BOS, 위 CHoCH, 아래 BOS, 아래 CHoCH) 날마다 True/False"""
    T = len(c)
    ub, uc, db, dc = (np.zeros(T, bool) for _ in range(4))
    os_ = 0
    top = bot = None              # [값, 넘었나]
    trend = 0
    for t in range(size, T):
        if np.isnan(c[t]) or np.isnan(h[t - size]):
            continue
        win_h, win_l = h[t - size + 1:t + 1], l[t - size + 1:t + 1]
        if np.isnan(win_h).any():
            continue
        prev = os_
        if h[t - size] > win_h.max():
            os_ = 0
        elif l[t - size] < win_l.min():
            os_ = 1
        if os_ == 0 and prev != 0:
            top = [h[t - size], False]
        if os_ == 1 and prev != 1:
            bot = [l[t - size], False]
        if top and not top[1] and c[t] > top[0] and c[t - 1] <= top[0]:
            (uc if trend == -1 else ub)[t] = True
            top[1], trend = True, 1
        if bot and not bot[1] and c[t] < bot[0] and c[t - 1] >= bot[0]:
            (dc if trend == 1 else db)[t] = True
            bot[1], trend = True, -1
    return ub, uc, db, dc


# ── 검증 ──
def trade(H, L, C, i, t0, hold=30):
    e = C[i, t0]
    for k in range(1, hold + 1):
        t = t0 + k
        if t >= C.shape[1] or np.isnan(C[i, t]):
            prev = C[i, t - 1] if not np.isnan(C[i, t - 1]) else e
            return prev / e - 1
        if L[i, t] <= e * (1 - STOP / 100):
            return -STOP / 100
        if H[i, t] >= e * (1 + TAKE / 100):
            return TAKE / 100
    return C[i, t0 + hold] / e - 1


def test_entry(data, S, pick):
    mk, days, H, L, C, V, first, binance = data
    mid = len(days) // 2 + 15
    sig, rnd, f, b = [], [], [[], []], [[], []]
    for i, t in zip(*np.nonzero(S)):
        if t < 31 or t >= len(days) - 46 or t - first[i] < 5:
            continue
        if pick == 'spike' and not spike_prone(i, t, mk, C, V, first, binance, 0):
            continue
        cand = [u for u in range(t - 15, t + 16) if not np.isnan(C[i, u]) and u - first[i] >= 5]
        u = cand[RNG.integers(len(cand))]
        a, r = trade(H, L, C, i, t) - 2 * COST, trade(H, L, C, i, u) - 2 * COST
        sig.append(a); rnd.append(r)
        (f if t < mid else b)[0].append(a)
        (f if t < mid else b)[1].append(r)
    if not sig:
        return None
    m = lambda x: float(np.mean(x) * 100) if x else float('nan')   # noqa: E731
    return {'n': len(sig), 'sig': m(sig), 'rand': m(rnd), 'win': float((np.array(sig) > 0).mean() * 100),
            'f_edge': m(f[0]) - m(f[1]), 'b_edge': m(b[0]) - m(b[1])}


def test_exit(data, S, hold, pick):
    mk, days, H, L, C, V, first, binance = data
    starts = list(range(30, len(days) - hold, 7))
    pos = [(t0, i) for t0 in starts for i in range(len(mk)) if not np.isnan(C[i, t0]) and t0 - first[i] >= 5
           and (pick == 'all' or spike_prone(i, t0, mk, C, V, first, binance, 0))]

    def run(mode, p=0.0):
        rets, early = [], 0
        for t0, i in pos:
            e = C[i, t0]
            px, how = e, 'end'
            for k in range(1, hold + 1):
                t = t0 + k
                if np.isnan(C[i, t]):
                    break
                if L[i, t] <= e * (1 - STOP / 100):
                    px, how = e * (1 - STOP / 100), 'stop'
                    break
                if H[i, t] >= e * (1 + TAKE / 100):
                    px, how = e * (1 + TAKE / 100), 'take'
                    break
                px = C[i, t]
                if (mode == 'sig' and S[i, t]) or (mode == 'rand' and RNG.random() < p):
                    how = 'early'
                    break
            rets.append((px * (1 - COST)) / (e * (1 + COST)) - 1)
            early += how == 'early'
        return float(np.mean(rets) * 100), early / len(pos) * 100
    base, _ = run('base')
    sig, se = run('sig')
    p = 0.01
    for _ in range(8):                                   # 아무 날 대조가 같은 비율로 팔게 맞춤
        rnd, re = run('rand', p)
        if se == 0 or abs(re - se) / se < 0.06:
            break
        p *= se / max(re, 0.1)
    return {'base': base, 'sig': sig, 'rand': rnd, 'early': se, 'rand_early': re}


def main():
    data = load()
    mk, days, H, L, C = data[0], data[1], data[2], data[3], data[4]
    K, T = C.shape
    print(f"코인 {K} · {days[0]} ~ {days[-1]}")
    sq_on, sq_off, val = (np.zeros((K, T), bool), np.zeros((K, T), bool), np.full((K, T), np.nan))
    smc = {k: np.zeros((K, T), bool) for k in ('iub', 'iuc', 'idb', 'idc', 'sub', 'suc', 'sdb', 'sdc')}
    for i in range(K):
        sq_on[i], sq_off[i], val[i] = squeeze(H[i], L[i], C[i])
        for pre, size in (('i', 5), ('s', 50)):
            ub, uc, db, dc = structure(H[i], L[i], C[i], size)
            smc[pre + 'ub'][i], smc[pre + 'uc'][i], smc[pre + 'db'][i], smc[pre + 'dc'][i] = ub, uc, db, dc
    prev_on = np.concatenate([np.zeros((K, 1), bool), sq_on[:, :-1]], axis=1)
    pv = np.concatenate([np.full((K, 1), np.nan), val[:, :-1]], axis=1)
    fired = sq_off & prev_on
    with np.errstate(invalid='ignore'):
        buys = {
            'SQ① 스퀴즈 터짐 & 모멘텀 +': fired & (val > 0),
            'SQ② 터짐 & 모멘텀 + & 오르는 중': fired & (val > 0) & (val > pv),
            'SQ③ 모멘텀 0 위로': (val > 0) & (pv <= 0),
            'SQ④ 모멘텀 바닥에서 돌아섬 (− 인데 오르기 시작)': (val < 0) & (val > pv) & (np.concatenate([np.full((K, 1), np.nan), pv[:, :-1]], axis=1) >= pv),
            'SMC 위로 CHoCH (내부 5봉)': smc['iuc'], 'SMC 위로 BOS (내부 5봉)': smc['iub'],
            'SMC 위로 CHoCH (큰 흐름 50봉)': smc['suc'], 'SMC 위로 BOS (큰 흐름 50봉)': smc['sub'],
        }
        sells = {
            'SQ 모멘텀 0 아래로': (val < 0) & (pv >= 0),
            'SQ 터짐 & 모멘텀 −': fired & (val < 0),
            'SMC 아래로 CHoCH (내부 5봉)': smc['idc'], 'SMC 아래로 BOS (내부 5봉)': smc['idb'],
            'SMC 아래로 CHoCH (큰 흐름 50봉)': smc['sdc'],
        }
    out = {'buy': {}, 'sell': {}}
    print("\n== 사는 신호: 신호 날 사서 30일 (손절 15 · 익절 50) vs 같은 코인 앞뒤 15일 아무 날")
    for name, S in buys.items():
        for pick in ('spike', 'all'):
            r = test_entry(data, S, pick)
            out['buy'][f'{name}|{pick}'] = r
            if r:
                print(f"  {name:<36} {'급등잦던' if pick == 'spike' else '모든코인'}: {r['n']:5d}번 · 신호 {r['sig']:+6.2f}% (이김 {r['win']:.0f}%) | 아무 날 {r['rand']:+6.2f}%"
                      f" → 신호 몫 {r['sig'] - r['rand']:+6.2f} (앞 {r['f_edge']:+.1f} · 뒤 {r['b_edge']:+.1f})")
    print("\n== 파는 신호: 산 코인을 신호 날 종가에 팖 (손절 15 · 익절 50 위에) vs 같은 비율 아무 날")
    for name, S in sells.items():
        for hold in (30, 90):
            for pick in ('spike', 'all'):
                r = test_exit(data, S, hold, pick)
                out['sell'][f'{name}|{pick}_{hold}'] = r
                print(f"  {name:<28} {'급등잦던' if pick == 'spike' else '모든코인'} {hold}일: 기준 {r['base']:+6.2f}% | 신호 {r['sig']:+6.2f}% (먼저 팖 {r['early']:.0f}%)"
                      f" | 아무 날 {r['rand']:+6.2f}% ({r['rand_early']:.0f}%) → 신호 몫 {r['sig'] - r['rand']:+.2f}")
    (HERE / 'outputs' / 'probe31_squeeze_smc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
