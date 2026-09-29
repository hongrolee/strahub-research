"""probe14 — 내 보유 코인 묶음, 범위 위에서 팔고 아래서 되사기 (블로그 8편)

실제 계좌(업비트)의 보유 코인 묶음 수익률이 며칠 동안 −27% ~ −31% 사이를 오간 것을 보고,
"묶음 전체를 최근 범위의 위쪽에서 팔고 아래쪽에서 되사면 들고 있기보다 나았을까"를 확인한다.
(과거 시세로 계산만 함, 실제 주문 없음)

규칙: 최근 L(1분·15분·1일 봉 기준) 동안 묶음 지수의 최저~최고를 범위로 보고
      지금이 범위의 위 hi 이상이면 전부 팔고, 아래 lo 이하면 전부 다시 삼 (lo/hi = 10/90, 20/80, 30/70)
      판단은 끝난 봉으로, 체결은 다음 봉 끝 (일봉만 그날 종가 무렵)
비용: 사거나 팔 때마다 0.15% (수수료 0.05% + 불리한 체결 0.1%)
묶음: 2026-09-30 01:08 보유 코인을 그때 평가금 비율 그대로 (과거에는 그때 상장돼 있던 코인끼리 비율 맞춤)
      ※ 지금 들고 있는 '양'을 그대로 과거에 대면, 그 뒤 크게 떨어진 코인이 과거 묶음을 거의 다 차지해 버림

세 가지로 본다
  live : 계좌 수익률 1분 기록 4일 (2026-09-25 ~ 09-30), 사고판 효과를 뺀 묶음 지수
  year : 15분봉 1년 (2025-09-28 ~ 2026-09-28), 평가금 0.3% 이상 코인. 달마다 '앞 3달 1등 규칙'을 그 달에 대 봄
  five : 일봉 5년 (2021-04 ~ 2026-09), 보유 코인 전부. 해마다 '앞 1년 1등 규칙'을 그 해에 대 봄

데이터:
  data/holdings_20260930.json  — 계좌 기록에서 만든 비중·1분 묶음 지수 (저장소에 포함)
                                 봇의 pnl_history.db 가 있으면 --prepare 로 다시 만들 수 있음
  data/box_cache/   15분봉 (없으면 업비트 공개 시세로 받음, probe10 과 같은 모양)
  data/trend_cache/ 일봉   (없으면 받음, probe11 과 같은 모양)
실행:  python probe14_holdings_swing.py [--prepare pnl_history.db]
결과:  콘솔 표 + outputs/probe14_holdings_swing.json

과거 계산 결과이며 앞으로의 수익을 뜻하지 않습니다.
"""
import argparse
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HOLD = ROOT / 'data' / 'holdings_20260930.json'
CACHE15 = ROOT / 'data' / 'box_cache'
CACHE1D = ROOT / 'data' / 'trend_cache'
OUT = ROOT / 'outputs' / 'probe14_holdings_swing.json'
KST = timezone(timedelta(hours=9))
COST = 0.0015
MIN_SHARE = 0.003                      # year: 평가금 0.3% 이상 코인만 (나머지는 합쳐 4% 남짓)
END15 = datetime(2026, 9, 28, tzinfo=timezone.utc)
BANDS = [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]
LOOKS = {
    'live': [('1시간', 60), ('3시간', 180), ('6시간', 360), ('12시간', 720), ('24시간', 1440)],
    'year': [('3시간', 12), ('6시간', 24), ('12시간', 48), ('24시간', 96), ('2일', 192), ('3일', 288), ('7일', 672)],
    'five': [('3일', 3), ('5일', 5), ('10일', 10), ('20일', 20), ('40일', 40)],
}


# ── 데이터 ──
def prepare(db):
    """봇 기록(pnl_history.db) → 비중 + 1분 묶음 지수. 앞뒤 분 모두 들고 있고 매수금액이 그대로인 코인만으로 1분 변화율"""
    c = sqlite3.connect(db)
    by = {}
    for ts, cur, buy, val in c.execute("select ts, currency, buy, value from pnl_coin order by ts"):
        by.setdefault(ts, {})[cur] = (buy, val)
    tss = sorted(by)
    idx = [1.0]
    for a, b in zip(tss, tss[1:]):
        A, B = by[a], by[b]
        common = [k for k in A if k in B and abs(A[k][0] - B[k][0]) < 1 and A[k][1] > 0]
        va = sum(A[k][1] for k in common)
        vb = sum(B[k][1] for k in common)
        idx.append(idx[-1] * (vb / va if va else 1))
    tot = {ts: (buy, val, rate) for ts, buy, val, rate in c.execute("select ts, total_buy, total_value, rate from pnl_total")}
    last = tss[-1]
    out = {
        'at': datetime.fromtimestamp(last, KST).isoformat(),
        'invested': round(tot[last][0]), 'value': round(tot[last][1]), 'rate': round(tot[last][2], 2),
        'holdings': {k: round(v[1]) for k, v in sorted(by[last].items(), key=lambda kv: -kv[1][1])},
        # 계좌 수익률: 기록이 잠깐 빠져 투자한 돈이 크게 줄어든 분(9/27 02:36~02:41, 6분)은 비워 둠 (묶음 지수는 영향 없음)
        'minute': [[ts, round(i, 6), round(tot[ts][2], 3) if ts in tot and tot[ts][0] > 0.95 * tot[last][0] else None]
                   for ts, i in zip(tss, idx)],
    }
    HOLD.parent.mkdir(parents=True, exist_ok=True)
    HOLD.write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
    print('만듦:', HOLD, len(tss), '분')


def pacer_get():
    from probe9_spike_rebuy import Pacer, get
    p = Pacer(0.12)
    return lambda url, params: get(p, url, params)


def load_15m(coin, get):
    have = sorted(CACHE15.glob(f'KRW-{coin}_15m_365d_*.json'))
    if have:
        rows = json.loads(have[-1].read_text(encoding='utf-8'))
    else:
        print('  15분봉 받는 중', coin, flush=True)
        rows, to, start = {}, END15, END15 - timedelta(days=365)
        while to > start:
            chunk = get('https://api.upbit.com/v1/candles/minutes/15',
                        {'market': 'KRW-' + coin, 'count': 200, 'to': to.strftime('%Y-%m-%dT%H:%M:%SZ')})
            if not chunk:
                break
            for x in chunk:
                rows[x['candle_date_time_utc']] = {'t': x['candle_date_time_utc'], 'o': x['opening_price'], 'h': x['high_price'],
                                                   'l': x['low_price'], 'c': x['trade_price']}
            oldest = datetime.fromisoformat(chunk[-1]['candle_date_time_utc']).replace(tzinfo=timezone.utc)
            if oldest >= to:
                break
            to = oldest
        cut = start.strftime('%Y-%m-%dT%H:%M:%S')
        rows = [rows[k] for k in sorted(rows) if k >= cut]
        CACHE15.mkdir(parents=True, exist_ok=True)
        (CACHE15 / f'KRW-{coin}_15m_365d_{END15:%Y%m%d}.json').write_text(json.dumps(rows), encoding='utf-8')
    return {datetime.fromisoformat(r['t']).replace(tzinfo=timezone.utc).timestamp(): r['c'] for r in rows}


def load_1d(coin, get):
    have = sorted(CACHE1D.glob(f'KRW-{coin}_1d_2000d_*.json'))
    if have:
        rows = json.loads(have[-1].read_text(encoding='utf-8'))
    else:
        import probe11_trend_momentum as p11
        from probe9_spike_rebuy import Pacer
        print('  일봉 받는 중', coin, flush=True)
        rows = p11.fetch_days(Pacer(0.12), 'KRW-' + coin, 2000)
    return {datetime.fromisoformat(r['d']).replace(tzinfo=KST).timestamp(): r['c'] for r in rows if r['d'] <= '2026-09-28'}


def basket_index(series, weights, step):
    """코인별 종가 → 비중 고정 묶음 지수 (그때 상장돼 있던 코인끼리 비율 맞춤)"""
    coins = list(series)
    t0 = min(min(s) for s in series.values())
    t1 = max(max(s) for s in series.values())
    grid = np.arange(t0, t1 + 1, step)
    P = np.full((len(grid), len(coins)), np.nan)
    for j, k in enumerate(coins):
        s, last = series[k], np.nan
        for i, g in enumerate(grid):
            last = s.get(g, last)
            P[i, j] = last
    w = np.array([weights[k] for k in coins], float)
    ok = ~np.isnan(P[1:]) & ~np.isnan(P[:-1])
    rr = np.where(ok, P[1:] / np.where(ok, P[:-1], 1), 0)
    ww = np.where(ok, w, 0)
    sw = ww.sum(1)
    r = np.where(sw > 0, (rr * ww).sum(1) / np.where(sw > 0, sw, 1), 1.0)
    return grid, np.concatenate([[1.0], np.cumprod(r)])


# ── 규칙 ──
def simulate(idx, L, lo_q, hi_q, delay=1):
    """봉마다 (전략 log 등락 - 비용), 코인 들고 있었는지, 사고팔았는지"""
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(idx)
    lr = np.log(idx[1:] / idx[:-1])
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    win = sliding_window_view(idx, L)[:n - L]          # win[j] = idx[j:j+L] → j+L 시점에서 본 최근 L봉 (지금 봉 제외)
    lo[L:], hi[L:] = win.min(1), win.max(1)
    hold = want = True
    pos = np.ones(n - 1, bool)
    trade = np.zeros(n - 1, bool)
    for i in range(n - 1):
        if delay and want != hold:                      # 앞 봉에서 본 신호를 이번 봉 끝에 체결
            hold = want
            trade[i] = True
        if not np.isnan(lo[i]) and hi[i] > lo[i]:
            p = (idx[i] - lo[i]) / (hi[i] - lo[i])
            if hold and p >= hi_q:
                want = False
            elif not hold and p <= lo_q:
                want = True
        if not delay and want != hold:                  # 일봉: 그날 종가 무렵 바로
            hold = want
            trade[i] = True
        pos[i] = hold
    strat = np.where(pos, lr, 0.0) + np.where(trade, np.log(1 - COST), 0.0)
    return strat, pos, trade, lr


def pct(x):
    return round(float(np.exp(x) - 1) * 100, 2)


def run_rules(part, idx, delay):
    rows, res = [], {}
    for name, L in LOOKS[part]:
        for lo_q, hi_q in BANDS:
            strat, pos, trade, lr = simulate(idx, L, lo_q, hi_q, delay)
            f = float(pos.mean())
            key = f'{name} {int(lo_q * 100)}/{int(hi_q * 100)}'
            res[key] = (strat, pos, lr)
            rows.append({'rule': key, 'look': name, 'band': [lo_q, hi_q], 'ret': pct(strat.sum()),
                         'vs_hold': pct(strat.sum() - lr.sum()), 'in_market': round(f * 100, 1), 'trades': int(trade.sum()),
                         # 쉬었던 시간만큼 피한 몫을 뺀 값: 같은 시간만큼 아무 때나 들고 있었을 때와 비교
                         'vs_same_time': pct(strat.sum() - f * lr.sum())})
    return rows, res


def walk(res, keys, periods, pick):
    per = {k: {p: float(res[k][0][keys == p].sum() - res[k][2][keys == p].sum()) for p in periods} for k in res}
    out, tot, wins = [], 0.0, 0
    for i in range(pick, len(periods)):
        prev = periods[i - pick:i]
        best = max(res, key=lambda k: sum(per[k][p] for p in prev))
        ex = per[best][periods[i]]
        tot += ex
        wins += ex > 0
        out.append({'period': periods[i], 'rule': best, 'before': pct(sum(per[best][p] for p in prev)), 'then': pct(ex)})
    return {'rows': out, 'total': pct(tot), 'wins': wins, 'n': len(out),
            'by_rule': {k: [pct(per[k][p]) for p in periods] for k in res}, 'periods': periods}


def show(title, rows):
    print(f'\n{title}')
    print('  규칙          |   결과  | 들고 있기보다 | 들고 있던 시간 |  매매  | 쉬었던 몫 빼면')
    for r in rows:
        print(f"  {r['rule']:<12} | {r['ret']:+7.1f}% | {r['vs_hold']:+8.1f}%    | {r['in_market']:5.0f}%        | {r['trades']:5d}번 | {r['vs_same_time']:+7.1f}%")


# ── 세 가지 ──
def part_live(H):
    m = H['minute']
    ts = np.array([r[0] for r in m])
    idx = np.array([r[1] for r in m])
    print(f"[live] {datetime.fromtimestamp(ts[0], KST):%m-%d %H:%M} ~ {datetime.fromtimestamp(ts[-1], KST):%m-%d %H:%M}, {len(ts)}분")
    print(f'  그냥 들고 있기 {(idx[-1] - 1) * 100:+.2f}%')
    # 지나고 나서 고른 고정 선 (정답 보고 풀기)
    best = None
    lo_i, hi_i = idx.min(), idx.max()
    for s in np.arange(np.ceil(lo_i * 1000 + 5), np.floor(hi_i * 1000)) / 1000:
        for b in np.arange(np.floor(lo_i * 1000), np.floor(s * 1000) - 4) / 1000:
            hold, v, t = True, 1.0, 0
            for i in range(1, len(idx)):
                if hold:
                    v *= idx[i] / idx[i - 1]
                if hold and idx[i] >= s:
                    hold, v, t = False, v * (1 - COST), t + 1
                elif not hold and idx[i] <= b:
                    hold, v, t = True, v * (1 - COST), t + 1
            ex = v / idx[-1] - 1
            if best is None or ex > best[0]:
                best = (ex, s, b, t)
    print(f'  지나고 나서 고른 선: 들고 있기보다 {best[0] * 100:+.2f}% ({best[3]}번 매매)')
    rows, res = run_rules('live', idx, 1)
    show('  [최근 범위 규칙, 4일]', rows)
    half = len(idx) // 2
    halves = {}
    for k in res:
        strat, _, lr = res[k]
        halves[k] = [pct(strat[:half].sum() - lr[:half].sum()), pct(strat[half:].sum() - lr[half:].sum())]
    first = max(halves, key=lambda k: halves[k][0])
    print(f'  앞 절반 1등 {first}: 앞 {halves[first][0]:+.2f}% → 뒤 {halves[first][1]:+.2f}%')
    # 그래프용: 계좌 수익률 15분 간격
    chart = [[int(t), r] for t, _, r in m[::15] if r is not None]
    rates = [r[2] for r in m if r[2] is not None]
    return {'from': datetime.fromtimestamp(ts[0], KST).isoformat(), 'to': datetime.fromtimestamp(ts[-1], KST).isoformat(),
            'minutes': len(ts), 'hold': pct(np.log(idx[-1])), 'hindsight': {'vs_hold': round(best[0] * 100, 2), 'trades': best[3]},
            'rules': rows, 'halves': halves, 'half_at': datetime.fromtimestamp(ts[half], KST).isoformat(),
            'rate_min': min(rates), 'rate_max': max(rates),
            'chart': chart}


def part_year(H, get):
    tot = sum(H['holdings'].values())
    coins = [k for k, v in H['holdings'].items() if v / tot >= MIN_SHARE]
    series = {k: load_15m(k, get) for k in coins}
    series = {k: v for k, v in series.items() if v}
    share = sum(H['holdings'][k] for k in series) / tot * 100
    grid, idx = basket_index(series, H['holdings'], 900)
    print(f'\n[year] 코인 {len(series)}개 (평가금의 {share:.1f}%), '
          f'{datetime.fromtimestamp(grid[0], KST):%Y-%m-%d} ~ {datetime.fromtimestamp(grid[-1], KST):%Y-%m-%d}, 들고 있기 {(idx[-1] - 1) * 100:+.1f}%')
    rows, res = run_rules('year', idx, 1)
    show('  [1년 15분봉]', rows)
    keys = np.array([datetime.fromtimestamp(g, KST).strftime('%Y-%m') for g in grid[1:]])
    periods = sorted(set(keys))
    wf = walk(res, keys, periods, 3)
    print(f"  앞 3달 1등을 다음 달에: 들고 있기보다 {wf['total']:+.1f}%, 나았던 달 {wf['wins']}/{wf['n']}")
    # 그래프용: 하루 한 점, 100만 원이 어떻게 되었나
    day = 96
    curves = {'hold': [round(float(v), 4) for v in idx[::day]]}
    for k in ('12시간 20/80', '7일 20/80'):
        eq = np.concatenate([[0.0], np.cumsum(res[k][0])])
        curves[k] = [round(float(np.exp(v)), 4) for v in eq[::day]]
    dates = [datetime.fromtimestamp(g, KST).strftime('%Y-%m-%d') for g in grid[::day]]
    return {'coins': len(series), 'share': round(share, 1), 'from': dates[0], 'to': dates[-1], 'hold': pct(np.log(idx[-1])),
            'rules': rows, 'walk': wf, 'curves': curves, 'dates': dates}


def part_five(H, get):
    series = {k: load_1d(k, get) for k in H['holdings']}
    series = {k: v for k, v in series.items() if v}
    tot = sum(H['holdings'].values())
    share = sum(H['holdings'][k] for k in series) / tot * 100
    grid, idx = basket_index(series, H['holdings'], 86400)
    print(f'\n[five] 코인 {len(series)}개 (평가금의 {share:.1f}%), '
          f'{datetime.fromtimestamp(grid[0], KST):%Y-%m-%d} ~ {datetime.fromtimestamp(grid[-1], KST):%Y-%m-%d}, 들고 있기 {(idx[-1] - 1) * 100:+.1f}%')
    rows, res = run_rules('five', idx, 0)
    show('  [5년 일봉]', rows)
    keys = np.array([datetime.fromtimestamp(g, KST).strftime('%Y') for g in grid[1:]])
    periods = sorted(set(keys))
    wf = walk(res, keys, periods, 1)
    print(f"  앞 1년 1등을 다음 해에: 들고 있기보다 {wf['total']:+.1f}%, 나았던 해 {wf['wins']}/{wf['n']}")
    return {'coins': len(series), 'share': round(share, 1), 'hold': pct(np.log(idx[-1])), 'rules': rows, 'walk': wf,
            'from': datetime.fromtimestamp(grid[0], KST).strftime('%Y-%m-%d'), 'to': datetime.fromtimestamp(grid[-1], KST).strftime('%Y-%m-%d')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', help='봇의 pnl_history.db 로 data/holdings_20260930.json 을 다시 만듦')
    ap.add_argument('--part', choices=['live', 'year', 'five', 'all'], default='all')
    args = ap.parse_args()
    if args.prepare:
        prepare(args.prepare)
    H = json.loads(HOLD.read_text(encoding='utf-8'))
    print(f"계좌 {H['at'][:16]}: 투자 {H['invested']:,}원 · 평가 {H['value']:,}원 · {H['rate']:+.2f}% · 코인 {len(H['holdings'])}개")
    get = pacer_get()
    out = {'account': {k: H[k] for k in ('at', 'invested', 'value', 'rate')}, 'coins': len(H['holdings'])}
    if args.part in ('live', 'all'):
        out['live'] = part_live(H)
    if args.part in ('year', 'all'):
        out['year'] = part_year(H, get)
    if args.part in ('five', 'all'):
        out['five'] = part_five(H, get)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
    print('\n저장:', OUT)


if __name__ == '__main__':
    main()
