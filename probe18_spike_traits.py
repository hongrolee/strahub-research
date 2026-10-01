"""probe18: 순간 급등이 나는 코인의 특성 — 언제 · 튀기 전에 무슨 일 · 어떤 코인

    python probe18_fetch.py          # 먼저 1분봉 받기 (data/spike_cache_all)
    python probe18_spike_traits.py   # → outputs/probe18_spike_traits.json + 요약 출력

순간 급등 = 봇의 '순간 급등' 판정(trading_bot_template/bot_core/conditions.py _t_spike_pump)에서 '직전 1시간 +10%'만 뺀 것:
  방금 끝난 1분의 거래대금이 1천만 원 이상이면서 직전 60분 평균의 5배 이상, 그 1분 동안 +5% 이상.
  (빈 분은 거래대금 0으로 채운 1분 격자 위에서 계산 — 봇·spike_verify.py와 같음.)
  같은 코인에서 30분 안에 이어진 것은 한 건으로. 봇 기준(직전 1시간 +10%)에도 맞는 건은 bot=True.

비교 기준:
  - 언제: 같은 기간 아무 시각(코인·분을 고르게 뽑은 것)과 비교
  - 직전 신호: 같은 코인 · 같은 시간대(KST 시)에서 급등과 6시간 넘게 떨어진 시각(대조군)과 비교
  - 적중률: 모든 코인에서 15분마다 '그 신호가 있었을 때 다음 60분 안에 급등이 난 비율'
과거에 있었던 일의 통계이며 앞으로를 예측하지 않는다.
"""
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data' / 'spike_cache_all'
TREND = HERE / 'data' / 'trend_cache'
OUT = HERE / 'outputs'
KST = timezone(timedelta(hours=9))

MIN_VALUE, MULT, RISE, RISE_1H = 10_000_000, 5.0, 5.0, 10.0
# 달러·유로에 값이 묶인 코인 (급등할 수 없는 구조라 뺌 — 받을 때 빠지지 않은 것들)
PEGGED = {'KRW-PYUSD', 'KRW-RLUSD', 'KRW-USDE', 'KRW-EURC', 'KRW-USDG', 'KRW-USDT', 'KRW-USDC', 'KRW-USD1', 'KRW-USDS', 'KRW-DAI'}
MERGE = 30          # 분: 같은 코인 연속 급등을 한 건으로
DAY = 1440
RNG = np.random.default_rng(18)


# ── 데이터 ──
def window():
    w = json.loads((DATA / '_window.json').read_text(encoding='utf-8'))
    s, e = datetime.fromisoformat(w['start']), datetime.fromisoformat(w['end'])
    return int(s.timestamp()), int(e.timestamp())


START, END = window()
N = (END - START) // 60
MIN_KST_HOUR = ((START // 60 + 9 * 60) // 60)        # 0번째 분의 KST 시(통산)


def load(market):
    """1분 격자: o,h,l,c (빈 분은 직전 종가), v (빈 분 0), first (첫 거래 분 번호)"""
    z = np.load(DATA / f'{market}.npz')
    idx = ((z['t'] - START) // 60).astype(np.int64)
    ok = (idx >= 0) & (idx < N)
    idx = idx[ok]
    if not len(idx):
        return None
    c = np.full(N, np.nan)
    o, h, l, v = np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan), np.zeros(N)
    c[idx], o[idx], h[idx], l[idx], v[idx] = z['c'][ok], z['o'][ok], z['h'][ok], z['l'][ok], z['v'][ok]
    first = int(idx.min())
    # 빈 분: 직전 종가로
    filled = np.where(np.isnan(c), np.nan, np.arange(N))
    last = np.fmax.accumulate(np.nan_to_num(filled, nan=-1))
    prev_c = np.where(last >= 0, c[np.maximum(last, 0).astype(int)], np.nan)
    miss = np.isnan(c)
    c = np.where(miss, prev_c, c)
    o, h, l = np.where(miss, c, o), np.where(miss, c, h), np.where(miss, c, l)
    return {'o': o, 'h': h, 'l': l, 'c': c, 'v': v, 'first': first}


def detect(b):
    """급등 분 번호들 (30분 안 연속은 첫 분만), 봇 기준 여부, 배수·상승률"""
    v, c, o = b['v'], b['c'], b['o']
    cs = np.concatenate([[0.0], np.cumsum(v)])
    i = np.arange(N)
    avg = np.full(N, np.nan)
    avg[60:] = (cs[60:N] - cs[0:N - 60]) / 60.0
    with np.errstate(invalid='ignore', divide='ignore'):
        rise = (c / o - 1) * 100
        ago = np.full(N, np.nan)
        ago[60:] = c[:N - 60]
        rise1h = (c / ago - 1) * 100
        hit = (i >= b['first'] + 60) & (v >= MIN_VALUE) & (avg > 0) & (v >= MULT * avg) & (rise >= RISE)
    out, last = [], -10 ** 9
    for k in np.flatnonzero(hit):
        if k - last > MERGE:
            out.append(int(k))
        last = k                                       # 이어진 급등은 마지막 분 기준으로 30분을 셈
    return [{'i': k, 'mult': float(v[k] / avg[k]), 'rise': float(rise[k]),
             'rise1h': float(rise1h[k]) if np.isfinite(rise1h[k]) else None,
             'bot': bool(np.isfinite(rise1h[k]) and rise1h[k] >= RISE_1H)} for k in out]


def kst_dt(i):
    return datetime.fromtimestamp(START + i * 60, KST)


def pct(a, b):
    return (a / b - 1) * 100 if b and np.isfinite(a) and np.isfinite(b) and b > 0 else np.nan


def pre_features(b, i):
    """급등(또는 고른 시각) i 직전까지만 본 특징 — i 분은 안 봄"""
    if i < max(DAY, b['first'] + DAY) or i - 1 < 0:
        return None
    v, c, h, l = b['v'], b['c'], b['h'], b['l']
    base = v[i - DAY:i - 240].mean()                   # 평소: 그 전 하루(직전 4시간 빼고)의 분당 거래대금
    if base <= 0:
        return None
    p = c[i - 1]
    seg_h, seg_l = h[i - 240:i].max(), l[i - 240:i].min()
    return {
        'vr15': float(v[i - 15:i].mean() / base), 'vr60': float(v[i - 60:i].mean() / base),
        'vr240': float(v[i - 240:i].mean() / base),
        'r15': pct(p, c[i - 16]), 'r60': pct(p, c[i - 61]), 'r240': pct(p, c[i - 241]), 'r1d': pct(p, c[i - DAY]),
        'rng240': float((seg_h - seg_l) / p * 100) if p > 0 else np.nan,
        'idle60': float((v[i - 60:i] == 0).mean() * 100),       # 직전 1시간 중 거래가 없던 분 비율
        'value_1d': float(v[i - DAY:i].sum()),
    }


def after_features(b, i):
    c, h = b['c'], b['h']
    p = c[i - 1]
    f = {}
    for k in (30, 240):
        f[f'a{k}'] = pct(c[i + k], p) if i + k < N else np.nan
    f['peak60'] = pct(h[i:min(N, i + 60)].max(), p)
    return f


# ── 코인 정보 ──
def listing_days(market, b):
    """상장한 지 며칠: 일봉 캐시(최대 2000일)의 첫날, 없으면 이 기간 안에서 첫 거래(기간 중 상장) — 모르면 None"""
    end = datetime.fromtimestamp(END, KST)
    f = sorted(TREND.glob(f'{market}_1d_*.json'))
    if f:
        rows = json.loads(f[-1].read_text(encoding='utf-8'))
        if rows:
            return (end - datetime.fromisoformat(rows[0]['d']).replace(tzinfo=KST)).days, len(rows) >= 1999
    if b['first'] > 0:
        return (end - kst_dt(b['first'])).days, False
    return None, False


def binance_bases():
    f = HERE / 'data' / 'binance_bases.json'
    if f.exists():
        return set(json.loads(f.read_text(encoding='utf-8')))
    import requests
    info = requests.get('https://api.binance.com/api/v3/exchangeInfo', timeout=30).json()
    bases = sorted({s['baseAsset'] for s in info['symbols'] if s.get('status') == 'TRADING'})
    f.write_text(json.dumps(bases), encoding='utf-8')
    return set(bases)


def bucket(x, edges, labels):
    if x is None:
        return '알 수 없음'
    for e, lab in zip(edges, labels):
        if x < e:
            return lab
    return labels[-1]


def q(a, p):
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)])
    return float(np.percentile(a, p)) if len(a) else None


def summary(a):
    a = [x for x in a if x is not None and np.isfinite(x)]
    return {'n': len(a), 'p25': q(a, 25), 'median': q(a, 50), 'p75': q(a, 75)}


def main():
    OUT.mkdir(exist_ok=True)
    meta = {m['market']: m for m in json.loads((DATA / '_markets.json').read_text(encoding='utf-8'))['markets']}
    markets = sorted(m for m in meta if (DATA / f'{m}.npz').exists() and m not in PEGGED)
    bin_bases = binance_bases()
    btc = load('KRW-BTC')

    events, controls, coins = [], [], []
    hit_rows = []                                   # 적중률: (특징들, 다음 60분 안 급등 여부)
    for n_m, market in enumerate(markets, 1):
        b = load(market)
        if b is None:
            continue
        evs = detect(b)
        ev_idx = np.array([e['i'] for e in evs], dtype=np.int64)
        covered = (N - b['first']) / DAY
        for e in evs:
            i = e['i']
            f = pre_features(b, i) or {}
            f.update(after_features(b, i))
            e.update(f)
            e['market'] = market
            dt = kst_dt(i)
            e['hour'], e['weekday'], e['date'] = dt.hour, dt.weekday(), dt.strftime('%Y-%m-%d')
            e['btc_1h'] = pct(btc['c'][i - 1], btc['c'][i - 61]) if i > 61 else np.nan
            prev = ev_idx[(ev_idx < i) & (ev_idx >= i - DAY)]
            e['prev24'] = bool(len(prev))
            events.append(e)
            # 대조군: 같은 코인 · 같은 KST 시 · 급등에서 6시간 넘게 떨어진 시각 5개
            cand = np.arange(max(DAY, b['first'] + DAY), N - 240)
            if len(cand):
                hrs = (((START // 60) + cand + 9 * 60) // 60) % 24
                cand = cand[hrs == dt.hour]
                if len(ev_idx):
                    far = np.min(np.abs(cand[:, None] - ev_idx[None, :]), axis=1) > 360
                    cand = cand[far]
                for j in RNG.choice(cand, size=min(5, len(cand)), replace=False) if len(cand) else []:
                    cf = pre_features(b, int(j))
                    if cf:
                        cf['market'] = market
                        controls.append(cf)
        # 적중률용: 15분마다 (앞쪽 하루는 빼고)
        js = np.arange(max(DAY, b['first'] + DAY), N - 60, 15)
        nxt = np.zeros(N + 61, dtype=bool)
        for k in ev_idx:
            nxt[max(0, k - 60):k + 1] = True          # j 가 [k-60, k] 안이면 다음 60분 안에 급등
        for j in js:
            f = pre_features(b, int(j))
            if f:
                hit_rows.append((f['vr15'], f['vr60'], f['r60'], f['rng240'], f['idle60'], bool(nxt[j])))
        # 코인 하나
        val_day = np.add.reduceat(b['v'][b['first']:], np.arange(0, N - b['first'], DAY)) if N - b['first'] >= DAY else np.array([b['v'].sum()])
        price = float(np.nanmedian(b['c'][b['first']:]))
        lday, old = listing_days(market, b)
        base = market.split('-', 1)[1]
        coins.append({'market': market, 'name': meta[market].get('korean_name'), 'events': len(evs),
                      'bot_events': sum(e['bot'] for e in evs), 'days': covered,
                      'per30': len(evs) / covered * 30 if covered > 0 else 0,
                      'value_day_median': float(np.median(val_day)), 'price': price,
                      'listing_days': lday, 'listing_old': old,
                      'warning': bool(meta[market].get('warning')), 'caution': sorted(meta[market].get('caution') or {}),
                      'binance': base in bin_bases})
        if n_m % 40 == 0:
            print(f'{n_m}/{len(markets)} 코인 · 급등 {len(events)}건', flush=True)

    res = analyse(events, controls, coins, hit_rows)
    res['meta'] = {'window_kst': [kst_dt(0).strftime('%Y-%m-%d %H:%M'), kst_dt(N - 1).strftime('%Y-%m-%d %H:%M')],
                   'coins': len(coins), 'events': len(events), 'controls': len(controls), 'hit_rows': len(hit_rows),
                   'rule': {'min_value': MIN_VALUE, 'mult': MULT, 'rise_1m': RISE, 'merge_min': MERGE, 'bot_rise_1h': RISE_1H}}
    clean = lambda x: None if isinstance(x, float) and not np.isfinite(x) else x
    for e in events:
        for k, v in list(e.items()):
            e[k] = clean(v)
    res['events'] = events
    res['coins'] = coins
    (OUT / 'probe18_spike_traits.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    print(json.dumps({k: v for k, v in res.items() if k not in ('events', 'coins')}, ensure_ascii=False, indent=1, default=float)[:6000])


def analyse(events, controls, coins, hit_rows):
    res = {}
    half = N // 2
    # 최근 늘었나: 주별
    weeks = Counter()
    week_coins = defaultdict(set)
    for e in events:
        d = datetime.strptime(e['date'], '%Y-%m-%d')
        wk = (d - timedelta(days=d.weekday())).strftime('%m-%d')
        weeks[wk] += 1
        week_coins[wk].add(e['market'])
    res['weekly'] = [{'week': w, 'events': weeks[w], 'coins': len(week_coins[w])} for w in sorted(weeks)]

    # 언제: 시간대·요일 — 기준선은 모든 분이 고르게 (시마다 같은 비율)이므로 1/24, 1/7 과 비교
    hours = Counter(e['hour'] for e in events)
    days = Counter(e['weekday'] for e in events)
    n = len(events)
    res['hour'] = [{'hour': h, 'events': hours[h], 'share': hours[h] / n * 100 if n else 0} for h in range(24)]
    res['weekday'] = [{'weekday': d, 'events': days[d], 'share': days[d] / n * 100 if n else 0} for d in range(7)]
    for part, rng_ in (('hour_first_half', range(0, half)), ('hour_second_half', range(half, N))):
        hs = Counter(e['hour'] for e in events if e['i'] in rng_)
        m = sum(hs.values())
        res[part] = [hs[h] / m * 100 if m else 0 for h in range(24)]
    # 비트코인 직전 1시간
    res['btc_1h'] = summary([e['btc_1h'] for e in events])
    # 동시 급등: 같은 10분 칸
    slot = Counter(e['i'] // 10 for e in events)
    res['simultaneous'] = {'events_in_shared_10min': sum(c for c in slot.values() if c > 1),
                           'share': sum(c for c in slot.values() if c > 1) / n * 100 if n else 0,
                           'max_in_10min': max(slot.values()) if slot else 0}

    # 직전 신호: 급등 vs 대조군
    feats = ['vr15', 'vr60', 'vr240', 'r15', 'r60', 'r240', 'r1d', 'rng240', 'idle60', 'value_1d']
    ev_pre = [e for e in events if e.get('vr60') is not None]
    res['pre'] = {f: {'spike': summary([e.get(f) for e in ev_pre]), 'control': summary([c.get(f) for c in controls])} for f in feats}
    res['pre_split'] = {}
    for part, cond in (('first', lambda e: e['i'] < half), ('second', lambda e: e['i'] >= half)):
        sub = [e for e in ev_pre if cond(e)]
        res['pre_split'][part] = {f: summary([e.get(f) for e in sub])['median'] for f in feats}
    res['prev24_share'] = sum(e['prev24'] for e in events) / n * 100 if n else 0

    # 적중률: 신호 구간별 '다음 60분 안 급등' 비율
    if hit_rows:
        arr = np.array([r[:5] for r in hit_rows], dtype=float)
        lab = np.array([r[5] for r in hit_rows])
        base = lab.mean() * 100
        res['hit_base'] = base
        res['hit'] = {}
        for k, name, edges in ((0, 'vr15', [0.5, 1, 2, 3, 5, 10]), (1, 'vr60', [0.5, 1, 1.5, 2, 3, 5]),
                               (2, 'r60', [-3, -1, 0, 1, 3, 5, 10]), (3, 'rng240', [1, 2, 3, 5, 8, 12]),
                               (4, 'idle60', [10, 30, 50, 70, 90])):
            x = arr[:, k]
            rows = []
            lo = -np.inf
            for e in edges + [np.inf]:
                m = (x >= lo) & (x < e) & np.isfinite(x)
                if m.sum() >= 200:
                    rows.append({'from': None if lo == -np.inf else lo, 'to': None if e == np.inf else e,
                                 'n': int(m.sum()), 'rate': float(lab[m].mean() * 100)})
                lo = e
            res['hit'][name] = rows

    # 튄 뒤 (참고)
    res['after'] = {k: {'above_share': (lambda a: sum(x > 0 for x in a) / len(a) * 100 if a else None)(
        [e[k] for e in events if e.get(k) is not None and np.isfinite(e[k])]), **summary([e.get(k) for e in events])}
        for k in ('a30', 'a240', 'peak60')}

    # 어떤 코인
    tot = sum(c['events'] for c in coins)
    ranked = sorted(coins, key=lambda c: -c['events'])
    res['concentration'] = {'coins_with_event': sum(c['events'] > 0 for c in coins), 'coins': len(coins),
                            'top10_share': sum(c['events'] for c in ranked[:10]) / tot * 100 if tot else 0,
                            'top10': [(c['market'], c['name'], c['events']) for c in ranked[:10]]}

    def group(key, fn):
        g = defaultdict(list)
        for c in coins:
            g[fn(c)].append(c)
        return [{'group': k, 'coins': len(v), 'with_event_share': sum(c['events'] > 0 for c in v) / len(v) * 100,
                 'per30_mean': float(np.mean([c['per30'] for c in v])), 'events': sum(c['events'] for c in v)}
                for k, v in g.items()]

    val_edges = [1e9, 3e9, 1e10, 3e10, 1e11]
    val_labels = ['10억 미만', '10억~30억', '30억~100억', '100억~300억', '300억~1000억', '1000억 이상']
    res['by_value'] = sorted(group('value', lambda c: bucket(c['value_day_median'], val_edges, val_labels)),
                             key=lambda r: val_labels.index(r['group']))
    p_edges = [1, 10, 100, 1000, 10000]
    p_labels = ['1원 미만', '1~10원', '10~100원', '100~1천원', '1천~1만원', '1만원 이상']
    res['by_price'] = sorted(group('price', lambda c: bucket(c['price'], p_edges, p_labels)), key=lambda r: p_labels.index(r['group']))
    l_edges = [90, 365, 730, 1460]
    l_labels = ['3개월 미만', '3개월~1년', '1~2년', '2~4년', '4년 이상']
    res['by_listing'] = sorted(group('listing', lambda c: bucket(c['listing_days'], l_edges, l_labels)), key=lambda r: (l_labels + ['알 수 없음']).index(r['group']))
    res['by_warning'] = group('warning', lambda c: '유의·주의 표시 있음' if (c['warning'] or c['caution']) else '표시 없음')
    res['by_binance'] = group('binance', lambda c: '바이낸스에도 있음' if c['binance'] else '바이낸스에 없음')
    # 반복성: 앞 절반에 튄 코인이 뒤 절반에도 튀나
    first = {e['market'] for e in events if e['i'] < half}
    second = {e['market'] for e in events if e['i'] >= half}
    all_m = {c['market'] for c in coins}
    res['repeat'] = {'first_half_coins': len(first),
                     'again_share': len(first & second) / len(first) * 100 if first else None,
                     'others_share': len(second - first) / len(all_m - first) * 100 if all_m - first else None}
    return res


if __name__ == '__main__':
    main()
