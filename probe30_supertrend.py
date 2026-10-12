"""
probe30: Supertrend(트레이딩뷰 기본 ATR 10 · 배수 3, 그 밖에 7·2 / 14·3) 가 정말 잘 맞나? — 일봉 (signal_log.db, probe25 자료)
 1) 파는 신호: 손절 15 · 익절 50 깔고, 추세가 '내림'으로 바뀐 날 종가에 팖 (probe25 방식 7일마다 급등잦던/모든 코인 30·90일)
    대조: 같은 비율로 아무 날에 팖
 2) 사는 신호: 추세가 '오름'으로 바뀐 날 종가에 삼 → 30일 (손절 15 · 익절 50)
    대조: 같은 코인 · 같은 달 안에서 아무 날에 삼 (시장 흐름은 같게, 날짜만 무작위)
 3) 추세 따라가기: '오름'으로 바뀌면 사고 '내림'으로 바뀌면 팖 (손절 · 익절 없음) vs 같은 기간 아무 날에 사서 같은 날 수만큼 들고 있기
비용 한쪽 0.15%. 결과: outputs/probe30_supertrend.json
"""
import json
from pathlib import Path

import numpy as np

from probe25_bot3_settings import load, spike_prone, COST

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(3)
STOP, TAKE = 15, 50
PARAMS = [(10, 3.0), (7, 2.0), (14, 3.0)]


def supertrend(H, L, C, n, m):
    """코인마다 추세 방향 +1(오름) / −1(내림), 값 없으면 0 — 트레이딩뷰 Supertrend 와 같은 계산 (ATR = RMA)"""
    K, T = C.shape
    D = np.zeros((K, T), int)
    for i in range(K):
        h, l, c = H[i], L[i], C[i]
        ok = np.flatnonzero(~np.isnan(c))
        if len(ok) < n + 2:
            continue
        a, b = ok[0], ok[-1]
        tr = np.full(T, np.nan)
        for t in range(a, b + 1):
            if np.isnan(c[t]):
                continue
            pc = c[t - 1] if t > a and not np.isnan(c[t - 1]) else c[t]
            tr[t] = max(h[t] - l[t], abs(h[t] - pc), abs(l[t] - pc))
        atr = np.full(T, np.nan)
        atr[a + n - 1] = np.nanmean(tr[a:a + n])
        for t in range(a + n, b + 1):
            atr[t] = atr[t - 1] if np.isnan(tr[t]) else (atr[t - 1] * (n - 1) + tr[t]) / n
        up = dn = None
        d = 1
        for t in range(a + n - 1, b + 1):
            if np.isnan(c[t]) or np.isnan(atr[t]):
                continue
            hl2 = (h[t] + l[t]) / 2
            bu, bl = hl2 + m * atr[t], hl2 - m * atr[t]
            pc = c[t - 1] if not np.isnan(c[t - 1]) else c[t]
            nu = bu if up is None or bu < up or pc > up else up
            nl = bl if dn is None or bl > dn or pc < dn else dn
            if up is not None:
                if d == -1 and c[t] > up:
                    d = 1
                elif d == 1 and c[t] < dn:
                    d = -1
            up, dn = nu, nl
            D[i, t] = d
    return D


def test_exit(data, D, hold, pick):
    mk, days, H, L, C, V, first, binance = data
    starts = list(range(30, len(days) - hold, 7))
    mid = starts[len(starts) // 2]
    pos = [(t0, i) for t0 in starts for i in range(len(mk)) if not np.isnan(C[i, t0]) and t0 - first[i] >= 5
           and (pick == 'all' or spike_prone(i, t0, mk, C, V, first, binance, 0))]

    def run(rule, p=None):
        f, b, early = [], [], 0
        for t0, i in pos:
            e = C[i, t0]
            px, how = e, 'end'
            for k in range(1, hold + 1):
                t = t0 + k
                if np.isnan(C[i, t]):
                    how = 'gone'
                    break
                if L[i, t] <= e * (1 - STOP / 100):
                    px, how = e * (1 - STOP / 100), 'stop'
                    break
                if H[i, t] >= e * (1 + TAKE / 100):
                    px, how = e * (1 + TAKE / 100), 'take'
                    break
                px = C[i, t]
                if rule == 'st' and D[i, t] == -1 and D[i, t - 1] == 1:
                    how = 'early'
                    break
                if rule == 'rand' and RNG.random() < p:
                    how = 'early'
                    break
            r = (px * (1 - COST)) / (e * (1 + COST)) - 1
            (f if t0 < mid else b).append(r)
            early += how == 'early'
        n = len(f) + len(b)
        return {'avg': float(np.mean(f + b) * 100), 'front': float(np.mean(f) * 100), 'back': float(np.mean(b) * 100), 'early': early / n * 100}
    base = run('base')
    st = run('st')
    alive = sum(hold for _ in pos)
    p = st['early'] / 100 * len(pos) / alive * 1.6          # 대략 같은 비율이 되게 (손절 · 익절로 일찍 끝나는 몫 보정)
    rd = run('rand', p)
    return base, st, rd


def test_entry(data, D, pick):
    """오름으로 바뀐 날 삼 → 30일 (손절 15 · 익절 50) vs 같은 코인 · 그 앞뒤 15일 안 아무 날"""
    mk, days, H, L, C, V, first, binance = data
    hold = 30

    def trade(i, t0):
        e = C[i, t0]
        for k in range(1, hold + 1):
            t = t0 + k
            if t >= C.shape[1] or np.isnan(C[i, t]):
                return (C[i, t - 1] if not np.isnan(C[i, t - 1]) else e) / e - 1
            if L[i, t] <= e * (1 - STOP / 100):
                return -STOP / 100
            if H[i, t] >= e * (1 + TAKE / 100):
                return TAKE / 100
        return C[i, t0 + hold] / e - 1
    sig, rnd = [], []
    mid = len(days) // 2 + 15
    fs, bs, fr, br = [], [], [], []
    for i in range(len(mk)):
        for t in range(31, len(days) - hold - 16):
            if D[i, t] == 1 and D[i, t - 1] == -1 and t - first[i] >= 5:
                if pick == 'spike' and not spike_prone(i, t, mk, C, V, first, binance, 0):
                    continue
                cand = [u for u in range(t - 15, t + 16) if not np.isnan(C[i, u]) and u - first[i] >= 5]
                u = cand[RNG.integers(len(cand))]
                a = trade(i, t) - 2 * COST
                b = trade(i, u) - 2 * COST
                sig.append(a); rnd.append(b)
                (fs if t < mid else bs).append(a)
                (fr if t < mid else br).append(b)
    return {'n': len(sig), 'sig': float(np.mean(sig) * 100), 'rand': float(np.mean(rnd) * 100),
            'sig_f': float(np.mean(fs) * 100), 'sig_b': float(np.mean(bs) * 100),
            'rand_f': float(np.mean(fr) * 100), 'rand_b': float(np.mean(br) * 100),
            'win': float((np.array(sig) > 0).mean() * 100)}


def test_follow(data, D):
    """오름으로 바뀌면 사고 내림으로 바뀌면 팖 (모든 코인) vs 같은 코인 아무 날에 사서 같은 날 수만큼"""
    mk, days, H, L, C, V, first, binance = data
    sig, rnd, held = [], [], []
    for i in range(len(mk)):
        t = 31
        while t < len(days) - 1:
            if D[i, t] == 1 and D[i, t - 1] == -1 and t - first[i] >= 5:
                u = t + 1
                while u < len(days) - 1 and D[i, u] != -1 and not np.isnan(C[i, u]):
                    u += 1
                if np.isnan(C[i, u]):
                    u -= 1
                if u <= t:
                    t += 1
                    continue
                k = u - t
                sig.append(C[i, u] / C[i, t] - 1 - 2 * COST)
                cand = [s for s in range(31, len(days) - k) if not np.isnan(C[i, s]) and not np.isnan(C[i, s + k]) and s - first[i] >= 5]
                s = cand[RNG.integers(len(cand))]
                rnd.append(C[i, s + k] / C[i, s] - 1 - 2 * COST)
                held.append(k)
                t = u
            t += 1
    sig, rnd = np.array(sig), np.array(rnd)
    return {'n': len(sig), 'sig': float(sig.mean() * 100), 'rand': float(rnd.mean() * 100), 'win': float((sig > 0).mean() * 100),
            'median': float(np.median(sig) * 100), 'days': float(np.mean(held))}


def main():
    data = load()
    mk, days, H, L, C = data[0], data[1], data[2], data[3], data[4]
    print(f"코인 {len(mk)} · {days[0]} ~ {days[-1]}")
    out = {}
    for n, m in PARAMS:
        D = supertrend(H, L, C, n, m)
        key = f'{n}x{m:g}'
        print(f"\n######## Supertrend ATR {n} · 배수 {m:g}  (오름 날 비율 {(D[:, 30:] == 1).sum() / max(1, (D[:, 30:] != 0).sum()) * 100:.0f}%)")
        res = {'exit': {}, 'entry': {}}
        print("  1) 파는 신호 (손절 15 · 익절 50 위에 '내림으로 바뀌면 팖')")
        for hold in (30, 90):
            for pick in ('spike', 'all'):
                base, st, rd = test_exit(data, D, hold, pick)
                res['exit'][f'{pick}_{hold}'] = {'base': base, 'st': st, 'rand': rd}
                print(f"    {'급등잦던' if pick == 'spike' else '모든코인'} {hold}일: 기준 {base['avg']:+6.2f}% (앞 {base['front']:+.1f} 뒤 {base['back']:+.1f})"
                      f" | Supertrend {st['avg']:+6.2f}% (앞 {st['front']:+.1f} 뒤 {st['back']:+.1f}, 먼저 팖 {st['early']:.0f}%)"
                      f" | 아무 날 {rd['avg']:+6.2f}% (먼저 팖 {rd['early']:.0f}%) → 신호 몫 {st['avg'] - rd['avg']:+.2f}")
        print("  2) 사는 신호 ('오름'으로 바뀐 날 사서 30일, 손절 15 · 익절 50)")
        for pick in ('spike', 'all'):
            e = test_entry(data, D, pick)
            res['entry'][pick] = e
            print(f"    {'급등잦던' if pick == 'spike' else '모든코인'}: {e['n']}번 · Supertrend {e['sig']:+6.2f}% (앞 {e['sig_f']:+.1f} 뒤 {e['sig_b']:+.1f}, 이김 {e['win']:.0f}%)"
                  f" | 같은 코인 앞뒤 15일 아무 날 {e['rand']:+6.2f}% (앞 {e['rand_f']:+.1f} 뒤 {e['rand_b']:+.1f}) → 신호 몫 {e['sig'] - e['rand']:+.2f}")
        f = test_follow(data, D)
        res['follow'] = f
        print(f"  3) 추세 따라가기 (오름에 사고 내림에 팖): {f['n']}번 · 평균 {f['sig']:+.2f}% (중앙 {f['median']:+.1f}, 이김 {f['win']:.0f}%, 평균 {f['days']:.0f}일 보유)"
              f" | 같은 날 수 아무 때나 {f['rand']:+.2f}% → 신호 몫 {f['sig'] - f['rand']:+.2f}")
        out[key] = res
    (HERE / 'outputs' / 'probe30_supertrend.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
