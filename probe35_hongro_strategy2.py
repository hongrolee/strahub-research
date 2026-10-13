"""
probe35: StraHub 'Hongro's Strategy2 (by aviva95)' — 원래 코드(AutoTrading_Flask/custom_indicators/hongro_strategy2.py)를 고치지 않고 그대로 돌림
- 자료: 15분봉 1년 74종을 1시간 · 4시간봉으로 묶음, 일봉 1년 253종(signal_log.db)
- 코인마다: 전략 수익 vs 같은 기간 그냥 들고 있기
- 공정한 비교(probe33 과 같은 생각): 거래 하나마다 (판 값 ÷ 산 값) − 같은 기간 다른 코인들의 평균 등락  → + 면 전략이 시장보다 나은 때 · 코인을 고름
- 앞 · 뒤 기간 따로 (거래를 산 시각으로 나눔). 비용: 전략 코드 안의 0.05% (미끄러짐 없음)
실행: AutoTrading_Flask 의 venv 파이썬 (talib 필요)
결과: outputs/probe35_hongro_strategy2.json
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, r'E:\MAYAI\AutoTrading_Flask')
from custom_indicators.hongro_strategy2 import hongro_strategy2_backtest  # noqa: E402
from probe32_intraday_ta import load_15m, to_hour  # noqa: E402
from probe33_ta_fair import daily_series, panel  # noqa: E402


def run(series, label):
    coins, grid, C, pos = panel(series)
    gidx = {int(t): k for k, t in enumerate(grid)}
    mid = (grid[0] + grid[-1]) // 2
    per, trades = [], []
    for j, cn in enumerate(coins):
        t, a = series[cn]
        if len(t) < 300:
            continue
        df = pd.DataFrame({'open': a[:, 0], 'high': a[:, 1], 'low': a[:, 2], 'close': a[:, 3], 'volume': a[:, 4]},
                          index=pd.to_datetime(t, unit='s'))
        with contextlib.redirect_stdout(io.StringIO()):
            r = hongro_strategy2_backtest(df)
        if not r.get('success'):
            continue
        start = 150 + 10
        hold = a[-1, 3] / a[min(start, len(a) - 1), 3] - 1
        per.append({'coin': cn, 'ret': r['total_return_pct'], 'hold': hold * 100, 'trades': r['total_trades'], 'win': r['win_rate'], 'mdd': r['max_drawdown']})
        buy = None
        for tr in r['trades']:
            ts = int(pd.Timestamp(tr['time']).timestamp())
            if tr['type'] == 'buy':
                buy = (ts, tr['price'], tr.get('condition'))
            elif buy:
                g0, g1 = gidx.get(buy[0]), gidx.get(ts)
                ret = tr['price'] / buy[1] - 1 - 0.001
                mk = None
                if g0 is not None and g1 is not None and g1 > g0:
                    with np.errstate(invalid='ignore', divide='ignore'):
                        m = C[:, g1] / C[:, g0] - 1
                    m = m[~np.isnan(m)]
                    if len(m) >= 5:
                        mk = float(m.mean())
                trades.append({'ret': ret, 'mk': mk, 'bars': (g1 - g0) if (g0 is not None and g1 is not None) else None,
                               'late': buy[0] >= mid, 'why': tr.get('condition'), 'cond': buy[2]})
                buy = None
    P = pd.DataFrame(per)
    T = pd.DataFrame(trades)
    T2 = T.dropna(subset=['mk'])
    edge = (T2['ret'] - T2['mk']) * 100
    out = {
        'coins': int(len(P)), 'avg_strategy': float(P['ret'].mean()), 'avg_hold': float(P['hold'].mean()),
        'beat_hold': float((P['ret'] > P['hold']).mean() * 100), 'median_strategy': float(P['ret'].median()),
        'avg_mdd': float(P['mdd'].mean()), 'trades': int(len(T)), 'win': float((T['ret'] > 0).mean() * 100),
        'avg_trade': float(T['ret'].mean() * 100), 'avg_bars': float(T['bars'].dropna().mean()),
        'edge': float(edge.mean()), 'edge_front': float(edge[~T2['late']].mean()), 'edge_back': float(edge[T2['late']].mean()),
        'edge_win': float((edge > 0).mean() * 100),
        'by_exit': {k: [int(len(g)), float(g['ret'].mean() * 100)] for k, g in T.groupby('why')},
        'by_entry': {k: [int(len(g)), float(((g['ret'] - g['mk']) * 100).mean())] for k, g in T2.groupby(T2['cond'].str.replace('+HC', '', regex=False))},
    }
    print(f"\n######## {label}")
    print(f"  코인 {out['coins']}개 · 전략 평균 {out['avg_strategy']:+.1f}% (중앙 {out['median_strategy']:+.1f}%) vs 그냥 들고 있기 {out['avg_hold']:+.1f}%"
          f" · 들고 있기보다 나은 코인 {out['beat_hold']:.0f}% · 최대 손실 폭 평균 {out['avg_mdd']:.1f}%")
    print(f"  거래 {out['trades']}번 · 이김 {out['win']:.0f}% · 거래당 평균 {out['avg_trade']:+.2f}% · 평균 {out['avg_bars']:.0f}봉 보유")
    print(f"  같은 기간 다른 코인 평균 대비: {out['edge']:+.2f}%p (앞 {out['edge_front']:+.2f} · 뒤 {out['edge_back']:+.2f}) · 시장보다 나은 거래 {out['edge_win']:.0f}%")
    print(f"  판 이유별 [번, 평균%]: {out['by_exit']}")
    print(f"  산 조건별 [번, 시장 대비 %p]: {out['by_entry']}")
    return out


def main():
    m15 = load_15m()
    out = {}
    h1 = {k: to_hour(t, a, 3600) for k, (t, a) in m15.items()}
    out['1h'] = run(h1, '1시간봉 · 1년 · 74종')
    h4 = {k: to_hour(t, a, 14400) for k, (t, a) in m15.items()}
    out['4h'] = run(h4, '4시간봉 · 1년 · 74종')
    d = daily_series()
    out['1d'] = run(d, f'일봉 · 1년 · {len(d)}종')
    (HERE / 'outputs' / 'probe35_hongro_strategy2.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
