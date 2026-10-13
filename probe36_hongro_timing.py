"""
probe36: Hongro's Strategy2 가 '하락을 알아채 밖에 있었나', 아니면 '늘 일부 시간 밖에 있었을 뿐'인가
- probe35 와 같은 원본 코드 · 자료(1시간 · 4시간봉 74종, 일봉 253종)
- 코인마다 전략이 들고 있던 봉(산 봉 다음 ~ 판 봉) / 밖에 있던 봉을 나눠, 봉 하나당 평균 등락을 비교
  → 하락을 알아챘다면 '밖에 있던 봉'의 평균이 '들고 있던 봉'보다 낮아야 함
- 대조: 같은 코인에서 들고 있던 봉 수만큼 아무 봉을 골라 들었다면 (200번 섞어 평균) — 같은 '시장에 있던 비율'
결과: outputs/probe36_hongro_timing.json
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
from probe33_ta_fair import daily_series  # noqa: E402

RNG = np.random.default_rng(36)


def run(series, label):
    rows = []
    for cn, (t, a) in series.items():
        if len(t) < 300:
            continue
        c = a[:, 3]
        df = pd.DataFrame({'open': a[:, 0], 'high': a[:, 1], 'low': a[:, 2], 'close': c, 'volume': a[:, 4]},
                          index=pd.to_datetime(t, unit='s'))
        with contextlib.redirect_stdout(io.StringIO()):
            r = hongro_strategy2_backtest(df)
        if not r.get('success'):
            continue
        pos = {int(x): k for k, x in enumerate(t)}
        held = np.zeros(len(c), bool)
        buy = None
        for tr in r['trades']:
            k = pos.get(int(pd.Timestamp(tr['time']).timestamp()))
            if k is None:
                continue
            if tr['type'] == 'buy':
                buy = k
            elif buy is not None:
                held[buy + 1:k + 1] = True              # 산 봉 다음 봉부터 판 봉까지의 등락을 받음
                buy = None
        lr = np.diff(np.log(c))                          # 봉 하나의 등락 (로그)
        h = held[1:]
        s = 160                                          # 원본이 판단을 시작하는 봉 뒤만
        lr, h = lr[s:], h[s:]
        if h.sum() < 5 or (~h).sum() < 5:
            continue
        rnd = [lr[RNG.choice(len(lr), h.sum(), replace=False)].sum() for _ in range(200)]
        rows.append({'coin': cn, 'exposure': h.mean(), 'in': lr[h].mean(), 'out': lr[~h].mean(),
                     'strat': lr[h].sum(), 'rand': float(np.mean(rnd)), 'hold': lr.sum()})
    R = pd.DataFrame(rows)
    pct = lambda x: (np.exp(x) - 1) * 100       # noqa: E731
    bar = lambda x: x * 100                     # noqa: E731
    out = {'coins': int(len(R)), 'exposure': float(R['exposure'].mean() * 100),
           'in_bar': float(bar(R['in'].mean())), 'out_bar': float(bar(R['out'].mean())),
           'out_lower_share': float((R['out'] < R['in']).mean() * 100),
           'strat': float(pct(R['strat']).mean()), 'rand_same_exposure': float(pct(R['rand']).mean()), 'hold': float(pct(R['hold']).mean()),
           'beat_rand_share': float((R['strat'] > R['rand']).mean() * 100)}
    print(f"\n######## {label}  (코인 {out['coins']}개, 비용 빼기 전)")
    print(f"  시장에 있던 시간 {out['exposure']:.0f}%")
    print(f"  봉 하나당 평균 등락: 들고 있던 봉 {out['in_bar']:+.3f}% · 밖에 있던 봉 {out['out_bar']:+.3f}%  (밖이 더 나빴던 코인 {out['out_lower_share']:.0f}%)")
    print(f"  그 기간 수익(코인 평균): 전략이 들고 있던 봉 {out['strat']:+.1f}% · 같은 시간만큼 아무 봉 {out['rand_same_exposure']:+.1f}% · 그냥 들고 있기 {out['hold']:+.1f}%"
          f"  (아무 봉보다 나은 코인 {out['beat_rand_share']:.0f}%)")
    return out


def main():
    m15 = load_15m()
    out = {'1h': run({k: to_hour(t, a, 3600) for k, (t, a) in m15.items()}, '1시간봉 · 1년 · 74종'),
           '4h': run({k: to_hour(t, a, 14400) for k, (t, a) in m15.items()}, '4시간봉 · 1년 · 74종')}
    d = daily_series()
    out['1d'] = run(d, f'일봉 · 1년 · {len(d)}종')
    (HERE / 'outputs' / 'probe36_hongro_timing.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
