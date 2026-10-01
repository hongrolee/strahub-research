"""
probe19: 블로그 11편 규칙(spike-prone-v1)을 probe18 1분봉 캐시(281종 · 90일)에 통째로 대 봄

StraHub 백테스트 엔진(AutoTrading_Flask/utils/spike_rule_backtest.py)을 그대로 불러 씀 → 웹 백테스트와 같은 계산.
코인마다 100만 원으로 따로 계산 (코인 하나만 보는 웹 백테스트와 같은 방식), 결과를 합쳐 글에 쓸 숫자를 만듦.
결과: outputs/probe19_rule_backtest.json
"""
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'AutoTrading_Flask'))

import probe18_spike_traits as p18  # noqa: E402
from utils import spike_rule_backtest as srb  # noqa: E402

OUT = HERE / 'outputs' / 'probe19_rule_backtest.json'


def facts_for(market):
    f = sorted(p18.TREND.glob(f'{market}_1d_*.json'))
    listed_ts, old = None, False
    if f:
        rows = json.loads(f[-1].read_text(encoding='utf-8'))
        if rows:
            old = len(rows) >= 1999
            listed_ts = int(datetime.fromisoformat(rows[0]['d']).replace(tzinfo=p18.KST).timestamp())
    base = market.split('-', 1)[1]
    return {'listed_ts': listed_ts, 'listed_old': old, 'on_binance': base in p18.binance_bases()}


def bars_for(market):
    b = p18.load(market)
    if b is None:
        return None
    first = b['first']
    out = []
    for k in range(first, p18.N):
        out.append({'t': p18.START + k * 60, 'o': float(b['o'][k]), 'h': float(b['h'][k]), 'l': float(b['l'][k]),
                    'c': float(b['c'][k]), 'v': float(b['v'][k])})
    return out


def main():
    tpl = srb.load_template('spike-prone-v1')
    s = srb.settings_from_template(tpl)
    markets = sorted(x.stem for x in p18.DATA.glob('KRW-*.npz') if x.stem not in p18.PEGGED)
    rows, all_trips = [], []
    for n, m in enumerate(markets, 1):
        bars = bars_for(m)
        if not bars or len(bars) < srb.DAY + 120:
            continue
        facts = facts_for(m)
        res = srb.simulate(bars, s, facts)
        sm = srb.summarize(res)
        rows.append({'market': m, **{k: sm[k] for k in ('return_pct', 'hold_return_pct', 'trades', 'closed', 'wins',
                                                         'stops', 'takes', 'spike_sets', 'max_drawdown_pct')}})
        for t in res['trips']:
            all_trips.append({'market': m, 'how': t['how'], 'pnl_pct': t['pnl_pct'], 'entry_t': t['entry_t'],
                              'hours': (t['exit_t'] - t['entry_t']) / 3600, 'why': t['why']})
        if n % 40 == 0:
            print(f'{n}/{len(markets)}', flush=True)

    traded = [r for r in rows if r['trades']]
    done = [t for t in all_trips if t['how'] != 'open']
    g = sorted(t['pnl_pct'] for t in done)

    def q(p):
        return g[min(len(g) - 1, int(p * len(g)))] if g else None

    out = {
        'meta': {'window': [p18.START, p18.END], 'coins': len(rows), 'settings': s},
        'coins_traded': len(traded),
        'trips': len(all_trips), 'closed': len(done),
        'win_share': sum(t['pnl_pct'] > 0 for t in done) / len(done) * 100 if done else None,
        'pnl_mean': sum(g) / len(g) if g else None, 'pnl_p25': q(0.25), 'pnl_median': q(0.5), 'pnl_p75': q(0.75),
        'how': {h: sum(t['how'] == h for t in all_trips) for h in ('stop', 'take', 'open')},
        'hours_median': sorted(t['hours'] for t in done)[len(done) // 2] if done else None,
        'coin_return_mean': sum(r['return_pct'] for r in traded) / len(traded) if traded else None,
        'coin_beat_hold_share': sum(r['return_pct'] > r['hold_return_pct'] for r in traded) / len(traded) * 100 if traded else None,
        'coin_positive_share': sum(r['return_pct'] > 0 for r in traded) / len(traded) * 100 if traded else None,
        'spike_sets': sum(r['spike_sets'] for r in rows),
        'by_why': {},
        'rows': rows,
        'trips_list': all_trips,
    }
    for t in done:
        d = out['by_why'].setdefault(t['why'], {'n': 0, 'win': 0, 'sum': 0.0})
        d['n'] += 1
        d['win'] += t['pnl_pct'] > 0
        d['sum'] += t['pnl_pct']
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in out.items() if k not in ('rows', 'trips_list')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
