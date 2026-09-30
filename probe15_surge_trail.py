"""probe15 — 하루 급등 뒤 꺾이면 일부 팔고 싸지면 되사기 (블로그 9편)

규칙: 15분봉 종가가 24시간 전보다 G% 이상 오르면 '하루 급등'
      → 그 뒤 48시간 안에, 그때까지 최고가에서 P% 떨어진 값에 닿으면 판 몫을 팜 (안 꺾이면 안 팜 — 셈에서 뺌)
      → 판 돈을 셋으로 나눠 판 값보다 d1·d2·d3% 싼 값에 닿으면 하나씩 되삼
      → H시간 안에 못 산 몫은 그때 종가에 삼
      결과 = 판 몫의 코인이 몇 % 늘었나(+)·줄었나(−). 사고팔 때 각 0.15% (수수료 0.05% + 불리한 체결 0.1%)
      같은 봉에서는 떨어짐을 먼저 봄(불리한 쪽). 봉이 이미 아래에서 시작하면 시가에 팜.
      한 번이 끝나기 전에는 같은 코인에서 새로 시작하지 않음.
비교: 같은 코인·같은 기간에서 '급등 없이 아무 때나' 시작해 똑같이 팔고 되사기 (10번 반복 평균)
      — 떨어지는 장에서는 그냥 팔고 되사기만 해도 코인이 늘기 때문

데이터:
  year : 15분봉 1년 (probe10 이 받은 data/box_cache + probe14 가 받은 보유 코인) — 앞 8개월로 고르고 뒤 4개월에 대 봄
  1m   : 1분봉 90일 (probe9 가 받은 data/spike_cache)
  HBAR : 2026-09-28 급등 1분봉 (data/hbar_1m_20260930.json, 저장소에 포함. 없으면 업비트에서 받음)
실행:  python probe15_surge_trail.py
결과:  콘솔 표 + outputs/probe15_surge_trail.json

과거 계산 결과이며 앞으로의 수익을 뜻하지 않습니다.
"""
import glob
import itertools
import json
import random
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE15 = ROOT / 'data' / 'box_cache'
CACHE1 = ROOT / 'data' / 'spike_cache'
HBAR = ROOT / 'data' / 'hbar_1m_20260930.json'
OUT = ROOT / 'outputs' / 'probe15_surge_trail.json'
KST = timezone(timedelta(hours=9))
COST = 0.0015
SPLIT = datetime(2026, 6, 1, tzinfo=KST).timestamp()
WAIT_H = 48
GRID = {'G': [15, 20, 30], 'P': [5, 8, 12], 'D': [(3, 6, 10), (5, 10, 15), (8, 12, 16)], 'H': [24, 72]}
DEFAULT = (20, 8, (5, 10, 15), 72)          # 봇 기본값 (이번 HBAR 같은 '천천히 오른 급등'에 맞춘 조합)


# ── 데이터 ──
def to_arrays(ts, ohlc, step):
    """[(시각, (o,h,l,c))] → 빈 칸 없는 배열 (거래 없던 칸은 직전 종가)"""
    t0 = ts[0]
    n = int((ts[-1] - t0) // step) + 1
    A = np.full((n, 4), np.nan)
    A[((np.array(ts) - t0) // step).astype(np.int64)] = np.array(ohlc, float)
    for k in range(1, n):
        if np.isnan(A[k, 0]):
            A[k] = A[k - 1, 3]
    return t0, A[:, 0].copy(), A[:, 1].copy(), A[:, 2].copy(), A[:, 3].copy()


def load_dir(pattern, step):
    out = {}
    for f in sorted(glob.glob(str(pattern))):
        raw = sorted(json.loads(Path(f).read_text(encoding='utf-8')), key=lambda r: r['t'])
        if not raw:
            continue
        ts = [datetime.fromisoformat(r['t']).replace(tzinfo=timezone.utc).timestamp() for r in raw]
        out[Path(f).name.split('_')[0][4:]] = to_arrays(ts, [(r['o'], r['h'], r['l'], r['c']) for r in raw], step)
    return out


def load_hbar():
    if not HBAR.exists():
        from probe9_spike_rebuy import Pacer, get
        p, rows = Pacer(0.15), {}
        to, start = datetime(2026, 9, 30, 7, 0, tzinfo=KST), datetime(2026, 9, 27, 12, 0, tzinfo=KST)
        while to > start:
            ch = get(p, 'https://api.upbit.com/v1/candles/minutes/1',
                     {'market': 'KRW-HBAR', 'count': 200, 'to': to.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')})
            if not ch:
                break
            for x in ch:
                rows[x['candle_date_time_kst']] = [x['opening_price'], x['high_price'], x['low_price'], x['trade_price']]
            to = datetime.fromisoformat(ch[-1]['candle_date_time_utc']).replace(tzinfo=timezone.utc)
        ks = sorted(rows)
        HBAR.write_text(json.dumps({'t': ks, 'b': [rows[k] for k in ks]}), encoding='utf-8')
    d = json.loads(HBAR.read_text(encoding='utf-8'))
    ts = [datetime.fromisoformat(t).replace(tzinfo=KST).timestamp() for t in d['t']]
    return to_arrays(ts, d['b'], 60)


# ── 한 번 ──
def episode(o, h, l, c, i0, P, D, H, bph, trace=False):
    """i0 봉 끝에서 시작. 반환 (끝난 봉, 결과% 또는 None[, 과정])"""
    n = len(c)
    peak, sell, si = c[i0], None, None
    for i in range(i0 + 1, min(n, i0 + 1 + WAIT_H * bph)):
        if l[i] <= peak * (1 - P / 100):
            sell, si = min(peak * (1 - P / 100), o[i]), i
            break
        peak = max(peak, h[i])
    if sell is None:
        return (min(n - 1, i0 + WAIT_H * bph), None) + (({'peak': peak},) if trace else ())
    got, coins, fills = sell * (1 - COST), 0.0, []
    left = [sell * (1 - d / 100) for d in D]
    end = min(n - 1, si + H * bph)
    for i in range(si + 1, end + 1):
        for t in list(left):
            if l[i] <= t:
                coins += got / 3 / (t * (1 + COST))
                left.remove(t)
                fills.append((i, t))
        if not left:
            end = i
            break
    done = not left
    if left:
        if si + H * bph > n - 1:                     # 자료가 끝나 기한까지 못 기다림 → 셈에서 뺌 (trace 면 진행 중으로)
            if trace:
                return n - 1, None, {'peak': peak, 'sell': (si, sell), 'fills': fills, 'left': left, 'open': True}
            return n - 1, None
        coins += got / 3 * len(left) / (c[end] * (1 + COST))
        fills += [(end, c[end])] * len(left)
    r = (coins - 1) * 100
    if trace:
        return end, r, {'peak': peak, 'sell': (si, sell), 'fills': fills, 'done': done}
    return end, r


def events(data, rule, bph, starts=None):
    t0, o, h, l, c = data
    G, P, D, H = rule
    step = 3600 // bph
    out = []
    if starts is None:
        k = 24 * bph
        up = np.zeros(len(c), bool)
        up[k:] = c[k:] >= c[:-k] * (1 + G / 100)
        i = k
        while i < len(c):
            if up[i]:
                end, r = episode(o, h, l, c, i, P, D, H, bph)
                if r is not None:
                    out.append((t0 + i * step, r))
                i = end + 1
            else:
                i += 1
    else:
        for i in starts:
            _, r = episode(o, h, l, c, i, P, D, H, bph)
            if r is not None:
                out.append((t0 + i * step, r))
    return out


def summ(rs):
    v = [r for _, r in rs]
    if not v:
        return {'n': 0}
    return {'n': len(v), 'avg': round(statistics.mean(v), 2), 'win': round(sum(x > 0 for x in v) / len(v) * 100, 1),
            'median': round(statistics.median(v), 2), 'worst': round(min(v), 1), 'best': round(max(v), 1)}


def fmt(s):
    if not s['n']:
        return '없음'
    return f"{s['n']:4d}번 · 코인 늘어남 {s['win']:4.0f}% · 평균 {s['avg']:+6.2f}% · 중간 {s['median']:+6.2f}% · 가장 나쁨 {s['worst']:+6.1f}%"


def name(r):
    return f'24시간 +{r[0]}% · 최고가에서 −{r[1]}% · 되사기 −{r[2][0]}/{r[2][1]}/{r[2][2]}% · {r[3]}시간'


# ── 세 가지 ──
def part_year():
    coins = load_dir(CACHE15 / '*_15m_365d_*.json', 900)
    print(f'[year] 15분봉 {len(coins)}종')
    rules = list(itertools.product(GRID['G'], GRID['P'], GRID['D'], GRID['H']))
    res = {r: [(t, x, k) for k, d in coins.items() for t, x in events(d, r, 4)] for r in rules}
    rows = []
    for r in rules:
        a = summ([(t, x) for t, x, _ in res[r] if t < SPLIT])
        b = summ([(t, x) for t, x, _ in res[r] if t >= SPLIT])
        rows.append({'rule': [r[0], r[1], list(r[2]), r[3]], 'front': a, 'back': b})
    both = sum(1 for x in rows if x['front'].get('avg', -1) > 0 and x['back'].get('avg', -1) > 0)
    print(f'  규칙 {len(rules)}개 중 두 기간 모두 평균 플러스: {both}개')

    def score(r):
        v = [x for t, x, _ in res[r] if t < SPLIT]
        return statistics.mean(v) / (statistics.pstdev(v) or 1) * len(v) ** 0.5 if len(v) >= 20 else -9
    top = max(rules, key=score)
    out = {'coins': len(coins), 'rules': rows, 'both_plus': both, 'top': [top[0], top[1], list(top[2]), top[3]]}
    for key, rule in (('top_rule', top), ('default', DEFAULT)):
        ev = res[rule]
        print(f'\n  [{key}] {name(rule)}')
        blk = {}
        random.seed(1)
        cnt = {}
        for t, x, k in ev:
            cnt.setdefault(k, [0, 0])[0 if t < SPLIT else 1] += 1
        for part, pname in ((0, 'front'), (1, 'back')):
            real = [(t, x) for t, x, _ in ev if (t < SPLIT) == (part == 0)]
            rnd = []
            for _ in range(10):
                for k, m in cnt.items():
                    t0, o, h, l, c = coins[k]
                    sp = int((SPLIT - t0) // 900)
                    lo, hi = (96, max(97, sp)) if part == 0 else (max(96, sp), len(c) - 1)
                    if m[part] and hi > lo:
                        rnd += events(coins[k], rule, 4, [random.randrange(lo, hi) for _ in range(m[part])])
            blk[pname] = summ(real)
            blk[pname + '_random'] = summ(rnd)
            print(f"    {'앞 8개월' if part == 0 else '뒤 4개월'}: 급등 뒤  {fmt(blk[pname])}")
            print(f"    {'앞 8개월' if part == 0 else '뒤 4개월'}: 아무 때나 {fmt(blk[pname + '_random'])}")
        days = {}
        for t, x, _ in ev:
            days.setdefault(datetime.fromtimestamp(t, KST).strftime('%Y-%m-%d'), []).append(x)
        dm = [statistics.mean(v) for v in days.values()]
        blk['days'] = {'n': len(days), 'avg': round(statistics.mean(dm), 2), 'win': round(sum(x > 0 for x in dm) / len(dm) * 100, 1),
                       'top': [[d, len(v), round(statistics.mean(v), 2)] for d, v in sorted(days.items(), key=lambda kv: -len(kv[1]))[:5]]}
        print(f"    날짜 {len(days)}일로 묶으면 하루 평균 {blk['days']['avg']:+.2f}% · 플러스인 날 {blk['days']['win']:.0f}%")
        v = sorted(x for _, x, _ in ev)
        blk['hist'] = [sum(1 for x in v if lo <= x < hi) for lo, hi in
                       ((-100, -20), (-20, -10), (-10, -5), (-5, 0), (0, 5), (5, 10), (10, 15), (15, 100))]
        out[key] = blk
    return out


def part_1m():
    coins = load_dir(CACHE1 / '*_90d_*.json', 60)
    print(f'\n[1m] 1분봉 90일 {len(coins)}종')
    out = {'coins': len(coins), 'rules': []}
    for rule in (DEFAULT, (20, 5, (5, 10, 15), 72), (15, 5, (3, 6, 10), 72), (15, 8, (3, 6, 10), 24)):
        s = summ([e for d in coins.values() for e in events(d, rule, 60)])
        out['rules'].append({'rule': [rule[0], rule[1], list(rule[2]), rule[3]], **s})
        print(f'  {name(rule)}: {fmt(s)}')
    return out


def part_hbar():
    t0, o, h, l, c = load_hbar()
    f = lambda i: datetime.fromtimestamp(t0 + i * 60, KST).isoformat(timespec='minutes')
    out = {'from': f(0), 'to': f(len(c) - 1), 'now': float(c[-1]), 'high': float(h.max()), 'high_at': f(int(h.argmax())),
           'series': [[f(i), float(c[i])] for i in range(0, len(c), 10)], 'rules': []}
    print(f"\n[HBAR] {out['from']} ~ {out['to']}, 최고 {out['high']:g} ({out['high_at']}), 지금 {out['now']:g}")
    for rule in (DEFAULT, (15, 5, (3, 6, 10), 72)):
        G, P, D, H = rule
        idx = [i for i in range(1440, len(c)) if c[i] >= c[i - 1440] * (1 + G / 100)]
        if not idx:
            continue
        i0 = idx[0]
        _, r, tr = episode(o, h, l, c, i0, P, D, H, 60, trace=True)
        item = {'rule': [G, P, list(D), H], 'start': [f(i0), float(c[i0])], 'peak': float(tr['peak'])}
        if 'sell' in tr:
            si, sp = tr['sell']
            item['sell'] = [f(si), round(float(sp), 2)]
            item['fills'] = [[f(i), round(float(p), 2)] for i, p in tr['fills']]
            item['left'] = [round(float(x), 2) for x in tr.get('left', [])]
            if r is None:   # 진행 중: 남은 몫을 지금 값에 산다고 치면
                got = sp * (1 - COST)
                coins = sum(got / 3 / (p * (1 + COST)) for _, p in tr['fills']) + got / 3 * len(tr['left']) / (c[-1] * (1 + COST))
                item['now_change'] = round((coins - 1) * 100, 2)
            else:
                item['change'] = round(r, 2)
        out['rules'].append(item)
        print(f'  {name(rule)}: {item}')
    return out


def main():
    out = {'year': part_year(), '1m': part_1m(), 'hbar': part_hbar(), 'default': [DEFAULT[0], DEFAULT[1], list(DEFAULT[2]), DEFAULT[3]]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
    print('\n저장:', OUT)


if __name__ == '__main__':
    main()
