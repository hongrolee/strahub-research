"""
probe40: probe39 를 7년 남짓으로 — 100만 원 하나로 '조건 맞으면 전부 사고, 파는 조건 맞으면 전부 팖'
- 자료: data/ohlcv_data_kst.db 일봉(고가 · 저가 · 종가 · 거래대금), 지금 업비트에 있는 코인 289개 (상장폐지된 코인은 없음 → 결과가 조금 좋게 나올 수 있음)
- 신호: StraBot signal_stats.scan_market 그대로 (조건마다 '처음 맞은 날', 기본 숫자) — Hongro 는 bot_core/strategy_hongro2.py
- 사고팔기 · 비용 · 대조: probe39 와 같음 (거래대금 가장 큰 코인 하나에 100%, 손절 15 · 익절 50 · 30일 / Hongro 손절 8 · 익절 25 · 고점 −8%,
  0.15%씩, 아무 날 아무 코인 같은 규칙 200번)
- 기간: 2019-01-01 ~ 자료 끝 전체 + 해마다 1월 1일에 100만 원으로 새로 시작
결과: outputs/probe40_all_in_long.json
"""
import importlib.util
import json
import sqlite3
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TBT = Path(r'E:\MAYAI\AutoTrading_Flask\trading_bot_template')
sys.path.insert(0, str(TBT))
DBL = HERE / 'data' / 'ohlcv_data_kst.db'
FEE = 0.0015
START_KRW = 1_000_000
START_DAY = '2019-01-01'
CONDS = ['desc_scallop', 'hs_top', 'double_top', 'streak_down', 'ma_cross_down', 'bb_lower']
STRATS = [
    ('시험: 하락 스캘럽 뒤 사기', ['desc_scallop'], 'cond'),
    ('시험: 헤드앤숄더 천정 뒤 사기', ['hs_top'], 'cond'),
    ('시험: 이중 천정 이탈 뒤 사기', ['double_top'], 'cond'),
    ('시험: 연속 하락 뒤 사기', ['streak_down'], 'cond'),
    ('시험: 평균선 하향 돌파 뒤 사기', ['ma_cross_down'], 'cond'),
    ('새 봇 2 규칙: 볼린저 아래로 벗어남', ['bb_lower'], 'cond'),
    ('Hongro 전략2 일봉', [], 'hongro'),
]


def load_long():
    con = sqlite3.connect(f'file:{DBL}?mode=ro', uri=True)
    tabs = [t for (t,) in con.execute("select name from sqlite_master where type='table'") if t.startswith('coin_KRW_')]
    data = {}
    for t in tabs:
        try:
            rows = con.execute(f'select timestamp, high_price, low_price, close_price, value_krw from "{t}" where interval_type=\'day\' order by timestamp').fetchall()
        except Exception:
            continue
        rows = [(datetime.fromtimestamp(ts).strftime('%Y-%m-%d'), h, l, c, v or 0.0) for ts, h, l, c, v in rows if c]
        if len(rows) >= 60:
            data['KRW-' + t[len('coin_KRW_'):]] = rows
    return data


def scan_one(args):
    from bot_core import signal_stats as ss
    market, rows = args
    return market, ss.scan_market((market, rows, 0, [], CONDS))


def hongro_one(args):
    spec = importlib.util.spec_from_file_location('h2', str(TBT / 'bot_core' / 'strategy_hongro2.py'))
    H2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H2)
    market, rows, start_day = args
    out = []
    for t in range(60, len(rows)):
        if rows[t][0] < start_day:
            continue
        w = rows[max(0, t - 119):t + 1]
        d = {'day_highs_long': [r[1] for r in w], 'day_lows_long': [r[2] for r in w], 'day_closes_long': [r[3] for r in w], 'day_values_long': [r[4] for r in w]}
        if H2.buy_test({}, {}, d):
            out.append(rows[t][0])
    return market, out


def main():
    data = load_long()
    mk = sorted(data)
    days = sorted({r[0] for rows in data.values() for r in rows})
    di = {d: k for k, d in enumerate(days)}
    K, T = len(mk), len(days)
    H, L, C, V = (np.full((K, T), np.nan) for _ in range(4))
    for i, m in enumerate(mk):
        for d, h, l, c, v in data[m]:
            j = di[d]
            H[i, j], L[i, j], C[i, j], V[i, j] = h, l, c, v
    first = np.array([np.argmax(~np.isnan(C[i])) for i in range(K)])
    print(f"코인 {K}개 · {days[0]} ~ {days[-1]} ({T}일)", flush=True)
    with Pool(6) as pool:
        sig = {}
        for market, rows in pool.imap_unordered(scan_one, [(m, data[m]) for m in mk], chunksize=4):
            i = mk.index(market)
            for cid, day, dr, entry in rows:
                if day in di:
                    sig.setdefault(cid, set()).add((i, di[day]))
        print('조건 신호 계산 끝', {c: len(v) for c, v in sig.items()}, flush=True)
        hsig = {}
        for market, ds in pool.imap_unordered(hongro_one, [(m, data[m], START_DAY) for m in mk], chunksize=4):
            i = mk.index(market)
            for d in ds:
                hsig.setdefault(di[d], []).append(i)
        print('Hongro 신호 계산 끝', sum(len(v) for v in hsig.values()), flush=True)
    spec = importlib.util.spec_from_file_location('h2', str(TBT / 'bot_core' / 'strategy_hongro2.py'))
    H2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H2)
    RNG = np.random.default_rng(40)

    def window(i, t):
        s = max(first[i], t - 119)
        return {'day_closes_long': list(C[i, s:t + 1]), 'day_highs_long': list(H[i, s:t + 1]),
                'day_lows_long': list(L[i, s:t + 1]), 'day_values_long': list(V[i, s:t + 1])}

    def run(daysig, kind, t_start, t_end, rand_p=None):
        cash, trades, curve, pos, t = float(START_KRW), [], [], None, t_start
        while t <= t_end:
            if pos is None:
                if rand_p is not None:
                    pick = None
                    if RNG.random() < rand_p:
                        cands = np.flatnonzero(~np.isnan(C[:, t]) & (t - first >= 5))
                        pick = int(cands[RNG.integers(len(cands))]) if len(cands) else None
                else:
                    cands = [i for i in daysig.get(t, []) if not np.isnan(C[i, t])]
                    pick = max(cands, key=lambda i: V[i, t]) if cands else None
                if pick is not None:
                    e = C[pick, t]
                    pos = (pick, t, e, cash * (1 - FEE) / e, e)
                    trades.append({'buy_day': days[t], 'coin': mk[pick][4:], 'buy': float(e)})
                    cash = 0.0
                curve.append(cash + (pos[3] * C[pos[0], t] if pos else 0))
                t += 1
                continue
            i, t0, e, qty, peak = pos
            px = why = None
            if np.isnan(C[i, t]):
                k = np.flatnonzero(~np.isnan(C[i, :t]))[-1]
                px, why = C[i, k], '거래 끝'
            elif kind == 'cond':
                if L[i, t] <= e * 0.85:
                    px, why = e * 0.85, '손절 15%'
                elif H[i, t] >= e * 1.5:
                    px, why = e * 1.5, '익절 50%'
                elif t - t0 >= 30:
                    px, why = C[i, t], '30일 지남'
            else:
                if L[i, t] <= e * 0.92:
                    px, why = e * 0.92, '손절 8%'
                elif H[i, t] >= e * 1.25:
                    px, why = e * 1.25, '익절 25%'
                elif peak > e and L[i, t] <= peak * 0.92:
                    px, why = peak * 0.92, '고점에서 8% 하락'
                else:
                    peak = max(peak, H[i, t])
                    r = H2.sell_test({'bot_entry': e}, {}, window(i, t))
                    if r:
                        px, why = C[i, t], 'Hongro 팔기'
            if px is not None:
                cash = qty * px * (1 - FEE)
                trades[-1].update({'sell_day': days[t], 'why': why, 'days': int(t - t0), 'ret': float(px / e - 1) * 100})
                pos = None
                curve.append(cash)
                t += 1
                continue
            pos = (i, t0, e, qty, peak)
            curve.append(qty * C[i, t])
            t += 1
        final = cash + (pos[3] * C[pos[0], t_end] if pos else 0)
        if pos:
            trades[-1].update({'sell_day': None, 'why': '아직 들고 있음', 'days': int(t_end - pos[1]), 'ret': float(C[pos[0], t_end] / pos[2] - 1) * 100})
        cv = np.array(curve)
        mdd = float(((cv - np.maximum.accumulate(cv)) / np.maximum.accumulate(cv)).min() * 100) if len(cv) else 0.0
        closed = [x for x in trades if x.get('sell_day')]
        return {'final': float(final), 'trades': len(trades), 'mdd': mdd, 'list': trades,
                'win': (sum(1 for x in closed if x['ret'] > 0) / len(closed) * 100) if closed else None,
                'busy_days': sum(x['days'] for x in trades)}

    def daysig_of(conds, kind):
        if kind == 'hongro':
            return hsig
        out = {}
        for i, t in sig.get(conds[0], set()):
            out.setdefault(t, []).append(i)
        return out

    def bench(t_start, t_end):
        btc = mk.index('KRW-BTC')
        b = START_KRW * (1 - FEE) ** 2 * C[btc, t_end] / C[btc, t_start]
        alive = [i for i in range(K) if not np.isnan(C[i, t_start])]
        ew = []
        for i in alive:
            k = np.flatnonzero(~np.isnan(C[i, :t_end + 1]))[-1]
            ew.append(C[i, k] / C[i, t_start])
        return b, START_KRW * (1 - FEE) ** 2 * float(np.mean(ew)), len(alive)

    periods = [('전체', START_DAY, days[-1])] + [(str(y), f'{y}-01-01', f'{y}-12-31') for y in range(2019, int(days[-1][:4]) + 1)]
    out = {'coins': K, 'data': [days[0], days[-1]], 'periods': {}}
    for pname, a, b in periods:
        t_start = next(k for k, d in enumerate(days) if d >= a)
        t_end = max(k for k, d in enumerate(days) if d <= b)
        btc, ew, n_alive = bench(t_start, t_end)
        print(f"\n=== {pname}: {days[t_start]} ~ {days[t_end]} · 그때 코인 {n_alive}개 · 비트코인 들고 있기 {btc:,.0f}원 · 모든 코인 같은 금액씩 {ew:,.0f}원", flush=True)
        rows = []
        for name, conds, kind in STRATS:
            ds = daysig_of(conds, kind)
            r = run(ds, kind, t_start, t_end)
            flat = max(1, (t_end - t_start + 1) - r['busy_days'])
            p = min(1.0, r['trades'] / flat)
            rnd = np.array([run({}, kind, t_start, t_end, rand_p=p)['final'] for _ in range(200 if pname == '전체' else 100)])
            r.update({'name': name, 'rand_final': float(rnd.mean()), 'rand_median': float(np.median(rnd)), 'beat': float((rnd < r['final']).mean() * 100)})
            rows.append(r)
            print(f"  {name:<28} → {r['final']:>13,.0f}원 · 거래 {r['trades']:>4}번 · 이김 {r['win'] or 0:4.0f}% · 최대 손실 폭 {r['mdd']:6.1f}%"
                  f" | 아무 날 아무 코인 중간값 {r['rand_median']:>13,.0f}원 · 이 전략이 이긴 비율 {r['beat']:3.0f}%", flush=True)
        out['periods'][pname] = {'from': days[t_start], 'to': days[t_end], 'btc': btc, 'ew': ew, 'coins': n_alive,
                                 'rows': [{k: v for k, v in r.items() if k != 'list'} | {'list': r['list'] if pname == '전체' else []} for r in rows]}
    (HERE / 'outputs' / 'probe40_all_in_long.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')


if __name__ == '__main__':
    main()
