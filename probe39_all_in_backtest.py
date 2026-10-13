"""
probe39: 100만 원 하나로 '조건 맞으면 전부 사고, 파는 조건 맞으면 전부 팖' — 시뮬레이션 봇 규칙들을 같은 판에서 (이홍로 님 확인용 표)
- 대상: 업비트 원화 코인 전부 (signal_log.db 일봉 · 기본 숫자 신호), 기간: 2026-01-03 ~ 신호 끝
- 한 번에 한 코인만: 들고 있지 않을 때 그날 신호가 뜬 코인 중 거래대금이 가장 큰 코인을 그날 종가에 100% 삼
- 팔기 ① 조건 봇(시험 봇 5개 · 볼린저): 손절 15 · 익절 50 (일봉 저가 · 고가로 먼저 닿으면 그 값) · 산 지 30일 지나면 종가
       ② Hongro 전략2 일봉: bot_core/strategy_hongro2.py 의 사기 · 팔기 + 손절 8 · 익절 25 · 고점(산 뒤 오른 때)에서 8% 내리면
- 비용: 사고팔 때마다 0.15% (수수료 0.05% + 미끄러짐 0.1%), 판 날 바로 다시 사지 않고 다음 날부터
- 비교: 비트코인 들고 있기 · 모든 코인 같은 금액씩 들고 있기 · 같은 규칙으로 '아무 날 아무 코인'(신호 대신 같은 빈도로 무작위, 200번 평균)
결과: outputs/probe39_all_in_backtest.json (거래 목록 포함)
"""
import importlib.util
import json
import sqlite3
from pathlib import Path

import numpy as np

from probe25_bot3_settings import load, DB

HERE = Path(__file__).resolve().parent
FEE = 0.0015
START_KRW = 1_000_000
RNG = np.random.default_rng(39)
spec = importlib.util.spec_from_file_location('h2', r'E:\MAYAI\AutoTrading_Flask\trading_bot_template\bot_core\strategy_hongro2.py')
H2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H2)

STRATS = [
    ('시험: 하락 스캘럽 뒤 사기', ['desc_scallop'], 'cond'),
    ('시험: 헤드앤숄더 천정 뒤 사기', ['hs_top'], 'cond'),
    ('시험: 이중 천정 이탈 뒤 사기', ['double_top'], 'cond'),
    ('시험: 연속 하락 뒤 사기', ['streak_down'], 'cond'),
    ('시험: 평균선 하향 돌파 뒤 사기', ['ma_cross_down'], 'cond'),
    ('새 봇 2 규칙: 볼린저 아래로 벗어남', ['bb_lower'], 'cond'),
    ('조합: 이중 천정 + 평균선 하향 돌파 (3일 안 둘 다)', ['double_top', 'ma_cross_down'], 'cond'),
    ('Hongro 전략2 일봉', [], 'hongro'),
]


def main():
    mk, days, H, L, C, V, first, binance = load()
    mi = {m: i for i, m in enumerate(mk)}
    di = {d: i for i, d in enumerate(days)}
    T = len(days)
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    sig = {}
    for cond, m, d, dr in con.execute('SELECT cond, market, day, dir FROM signals'):
        if m in mi and d in di:
            sig.setdefault(cond, set()).add((mi[m], di[d]))
    t_start = min(t for s in sig.values() for _, t in s)
    t_end = T - 1

    def day_signals(conds):
        """날짜 → 그날 신호 뜬 코인들"""
        if len(conds) == 1:
            pairs = sig[conds[0]]
        else:                                                    # 같은 코인에서 3일 안에 둘 다 → 늦게 뜬 날
            bi = {}
            for i, t in sig[conds[1]]:
                bi.setdefault(i, []).append(t)
            pairs = {(i, max(t, u)) for i, t in sig[conds[0]] for u in bi.get(i, []) if abs(u - t) <= 3}
        out = {}
        for i, t in pairs:
            out.setdefault(t, []).append(i)
        return out

    def window(i, t):
        s = max(0, t - 119)
        c = C[i, s:t + 1]
        if np.isnan(c).any():
            ok = ~np.isnan(c)
            k = np.flatnonzero(~ok)[-1] + 1 if (~ok).any() else 0
            s += k
        return {'day_closes_long': list(C[i, s:t + 1]), 'day_highs_long': list(H[i, s:t + 1]),
                'day_lows_long': list(L[i, s:t + 1]), 'day_values_long': list(V[i, s:t + 1])}

    def hongro_signals():
        out = {}
        for i in range(len(mk)):
            for t in range(max(t_start, first[i] + 60), t_end + 1):
                if np.isnan(C[i, t]):
                    continue
                w = window(i, t)
                if len(w['day_closes_long']) < 60:
                    continue
                if H2.buy_test({}, {}, w):
                    out.setdefault(t, []).append(i)
        return out

    def run(daysig, kind, rand_p=None):
        cash, trades, curve = float(START_KRW), [], []
        pos = None                                               # (i, t0, entry, qty, peak)
        t = t_start
        while t <= t_end:
            if pos is None:
                if rand_p is not None:
                    cands = [i for i in range(len(mk)) if not np.isnan(C[i, t]) and t - first[i] >= 5] if RNG.random() < rand_p else []
                    pick = cands[RNG.integers(len(cands))] if cands else None
                else:
                    cands = [i for i in daysig.get(t, []) if not np.isnan(C[i, t])]
                    pick = max(cands, key=lambda i: V[i, t]) if cands else None
                if pick is not None:
                    e = C[pick, t]
                    qty = cash * (1 - FEE) / e
                    pos = (pick, t, e, qty, e)
                    trades.append({'buy_day': days[t], 'coin': mk[pick].replace('KRW-', ''), 'buy': float(e)})
                    cash = 0.0
                curve.append(cash + (pos[3] * C[pos[0], t] if pos else 0))
                t += 1
                continue
            i, t0, e, qty, peak = pos
            if np.isnan(C[i, t]):                                # 상장폐지 등: 마지막 값에 정리
                px, why = C[i, t - 1], '거래 끝'
            else:
                px = why = None
                if kind == 'cond':
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
                    else:
                        if peak > e and L[i, t] <= peak * 0.92:
                            px, why = peak * 0.92, '고점에서 8% 하락'
                        else:
                            peak = max(peak, H[i, t])
                            r = H2.sell_test({'bot_entry': e}, {}, window(i, t))
                            if r:
                                px, why = C[i, t], 'Hongro 팔기: ' + r['s_why']
            if px is not None:
                cash = qty * px * (1 - FEE)
                trades[-1].update({'sell_day': days[t], 'sell': float(px), 'why': why, 'days': int(t - t0),
                                   'ret': float((px * (1 - FEE)) / (e / (1 - FEE)) - 1) * 100, 'after': round(cash)})
                pos = None
                curve.append(cash)
                t += 1
                continue
            pos = (i, t0, e, qty, peak)
            curve.append(qty * C[i, t])
            t += 1
        final = cash + (pos[3] * C[pos[0], t_end] if pos else 0)
        if pos:
            trades[-1].update({'sell_day': None, 'why': '아직 들고 있음', 'days': int(t_end - pos[1]),
                               'ret': float(C[pos[0], t_end] / pos[2] - 1) * 100, 'after': round(final)})
        cv = np.array(curve)
        mdd = float(((cv - np.maximum.accumulate(cv)) / np.maximum.accumulate(cv)).min() * 100) if len(cv) else 0.0
        closed = [x for x in trades if x.get('sell_day')]
        return {'final': round(final), 'ret': (final / START_KRW - 1) * 100, 'trades': len(trades), 'mdd': mdd,
                'win': (sum(1 for x in closed if x['ret'] > 0) / len(closed) * 100) if closed else None,
                'hold': float(np.mean([x['days'] for x in trades])) if trades else 0, 'list': trades,
                'in_market': None}

    btc = mi['KRW-BTC']
    btc_hold = START_KRW * (1 - FEE) ** 2 * C[btc, t_end] / C[btc, t_start]
    alive0 = [i for i in range(len(mk)) if not np.isnan(C[i, t_start])]
    ew = []
    for i in alive0:
        last = C[i, t_end] if not np.isnan(C[i, t_end]) else C[i, np.flatnonzero(~np.isnan(C[i]))[-1]]
        ew.append(last / C[i, t_start])
    ew_hold = START_KRW * (1 - FEE) ** 2 * float(np.mean(ew))

    results = []
    print(f"기간 {days[t_start]} ~ {days[t_end]} ({t_end - t_start + 1}일) · 시작 {START_KRW:,}원 · 비용 사고팔 때마다 0.15%")
    print(f"  비트코인 들고 있기 → {btc_hold:,.0f}원 · 모든 코인({len(alive0)}개) 같은 금액씩 들고 있기 → {ew_hold:,.0f}원")
    for name, conds, kind in STRATS:
        daysig = hongro_signals() if kind == 'hongro' else day_signals(conds)
        r = run(daysig, kind)
        # 대조: 같은 팔기 규칙으로, 신호 대신 아무 날 아무 코인 (들고 있지 않은 날 산 확률을 그 전략과 맞춤)
        flat_days = max(1, (t_end - t_start + 1) - sum(x['days'] for x in r['list']))
        p = min(1.0, r['trades'] / flat_days)
        rnd = [run({}, kind, rand_p=p)['final'] for _ in range(200)]
        r['rand_final'] = float(np.mean(rnd))
        r['rand_beat'] = float(np.mean(np.array(rnd) < r['final']) * 100)
        r.update({'name': name, 'conds': conds, 'kind': kind, 'signal_days': len(daysig)})
        results.append(r)
        print(f"  {name:<34} → {r['final']:>11,}원 ({r['ret']:+6.1f}%) · 거래 {r['trades']:>3}번 · 이김 {r['win'] or 0:4.0f}% · 평균 {r['hold']:4.1f}일"
              f" · 최대 손실 폭 {r['mdd']:6.1f}% | 아무 날 아무 코인 같은 규칙 평균 {r['rand_final']:,.0f}원 (이 전략이 이긴 비율 {r['rand_beat']:.0f}%)")
    out = {'period': [days[t_start], days[t_end]], 'start_krw': START_KRW, 'btc_hold': btc_hold, 'ew_hold': ew_hold, 'results': results}
    (HERE / 'outputs' / 'probe39_all_in_backtest.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')


if __name__ == '__main__':
    main()
