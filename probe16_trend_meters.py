"""probe16 — "지금 오르고 있나"를 재는 방법 7가지, 비트코인 일봉에서 언제 켜지고 서로 얼마나 겹치나 (블로그 10편)

잰 것 (모두 그날 종가까지로만 계산, 앞날은 계산하지 않음):
  RSI(14) > 70 · 볼린저 %b(20, 2) > 1 · 3일 연속 상승 · 종가 > 20일 평균 · 20일 평균 > 60일 평균
  · MACD(12, 26) > 신호선(9) · MACD > 0
본 것:
  - 켜져 있던 날의 비율, 한 번 켜지면 평균 며칠 이어졌나, 1년에 몇 번 켜졌다 꺼졌다 했나 (민감도)
  - 서로 얼마나 겹치나: A가 켜진 날 중 B도 켜진 비율
  - 마지막 날(2026-09-21)의 값, 최근 180일 동안 켜진 모습 · 최근 120일 캔들과 지표 선 (그래프용)
다음날 수익은 보지 않음 — 그것은 probe1~7(블로그 1~3편)에서 다룸

데이터: data/ohlcv_data_kst.db (fetch_data.py), KRW-BTC 일봉, 2026-09-21 까지 (1~3편과 같은 기간)
실행:  python probe16_trend_meters.py
결과:  콘솔 표 + outputs/probe16_trend_meters.json
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DB, TABLE, KST = ROOT / 'data' / 'ohlcv_data_kst.db', 'coin_KRW_BTC', 9 * 3600
END = '2026-09-21'
OUT = ROOT / 'outputs' / 'probe16_trend_meters.json'


def load():
    con = sqlite3.connect('file:' + str(DB) + '?mode=ro', uri=True)
    df = pd.read_sql_query("SELECT timestamp, open_price, high_price, low_price, close_price FROM " + TABLE
                           + " WHERE interval_type='day' ORDER BY timestamp", con)
    con.close()
    df['date'] = pd.to_datetime(df['timestamp'] - KST, unit='s', utc=True).dt.tz_localize(None).dt.normalize()
    df = df.drop_duplicates('date').sort_values('date')
    df = df[df['date'] <= END].reset_index(drop=True)
    df = df.rename(columns={'open_price': 'open', 'high_price': 'high', 'low_price': 'low', 'close_price': 'close'})
    return df[['date', 'open', 'high', 'low', 'close']]


def meters(d):
    c = d['close']
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d['rsi'] = 100 - 100 / (1 + gain / loss)
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    d['bb_up'], d['bb_lo'] = ma20 + 2 * sd20, ma20 - 2 * sd20
    d['pctb'] = (c - (ma20 - 2 * sd20)) / (4 * sd20)
    up = (delta > 0).astype(int)
    d['streak'] = up.groupby((delta <= 0).cumsum()).cumsum()
    d['ma20'], d['ma60'] = ma20, c.rolling(60).mean()
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    d['macd'] = ema12 - ema26
    d['signal'] = d['macd'].ewm(span=9, adjust=False).mean()
    on = pd.DataFrame({
        'rsi70': d['rsi'] > 70,
        'bb': d['pctb'] > 1,
        'streak3': d['streak'] >= 3,
        'above20': c > ma20,
        'ma20_60': ma20 > d['ma60'],
        'macd_sig': d['macd'] > d['signal'],
        'macd0': d['macd'] > 0,
    })
    return d, on


NAMES = {'rsi70': 'RSI(14) 70 초과', 'bb': '볼린저 %b 1 초과', 'streak3': '3일 연속 상승', 'above20': '종가 > 20일 평균',
         'ma20_60': '20일 평균 > 60일 평균', 'macd_sig': 'MACD > 신호선', 'macd0': 'MACD > 0'}


def other_coins(end=END, min_days=365):
    """업비트 원화 코인 일봉(probe11 이 받은 data/trend_cache)에 같은 7가지를 대 봄 (종가만 쓰므로 일봉 종가로 충분)"""
    rows = []
    for f in sorted((ROOT / 'data' / 'trend_cache').glob('KRW-*_1d_2000d_*.json')):
        coin = f.name.split('_')[0][4:]
        raw = [r for r in json.loads(f.read_text(encoding='utf-8')) if r['d'] <= end]
        if len(raw) < min_days + 60:
            continue
        d = pd.DataFrame({'date': pd.to_datetime([r['d'] for r in raw]), 'close': [r['c'] for r in raw]})
        d, on = meters(d)
        d, on = d.iloc[60:].reset_index(drop=True), on.iloc[60:].reset_index(drop=True)
        r = {'coin': coin, 'days': len(d), 'from': str(d['date'].iloc[0].date())}
        r['ret'] = round(float(d['close'].iloc[-1] / d['close'].iloc[0] - 1) * 100, 1)
        r['flat'] = round(float((d['close'].diff() == 0).mean() * 100), 1)   # 종가가 전날과 똑같았던 날 (호가 단위가 큰 저가 코인)
        for k in on:
            s = on[k].values
            flips = int(((s[1:]) & (~s[:-1])).sum() + s[0])
            r[k] = round(float(s.mean() * 100), 1)
            r[k + '_py'] = round(flips / (len(s) / 365.25), 1)
        r['rsi_in_above20'] = round(float(on.loc[on['rsi70'], 'above20'].mean() * 100), 1) if on['rsi70'].any() else None
        r['rsi_in_macd0'] = round(float(on.loc[on['rsi70'], 'macd0'].mean() * 100), 1) if on['rsi70'].any() else None
        r['bb_in_above20'] = round(float(on.loc[on['bb'], 'above20'].mean() * 100), 1) if on['bb'].any() else None
        r['above20_in_rsi'] = round(float(on.loc[on['above20'], 'rsi70'].mean() * 100), 1)
        r['all7'] = round(float((on.sum(axis=1) == 7).mean() * 100), 1)
        r['none'] = round(float((on.sum(axis=1) == 0).mean() * 100), 1)
        rows.append(r)
    return rows


def main():
    d = load()
    d, on = meters(d)
    ok = d.index >= 60                       # 60일 평균이 나오는 날부터
    d, on = d[ok].reset_index(drop=True), on[ok].reset_index(drop=True)
    n = len(d)
    years = n / 365.25
    print(f"KRW-BTC 일봉 {d['date'].iloc[0].date()} ~ {d['date'].iloc[-1].date()} ({n}일)")
    stats = {}
    for k in on:
        s = on[k].values
        runs, cur = [], 0
        for v in s:
            if v:
                cur += 1
            elif cur:
                runs.append(cur); cur = 0
        if cur:
            runs.append(cur)
        stats[k] = {'name': NAMES[k], 'share': round(s.mean() * 100, 1), 'runs': len(runs),
                    'avg_run': round(float(np.mean(runs)), 1), 'per_year': round(len(runs) / years, 1),
                    'longest': int(max(runs))}
        print(f"  {NAMES[k]:<16} 켜진 날 {s.mean()*100:5.1f}% · 한 번에 평균 {np.mean(runs):5.1f}일 · 1년에 {len(runs)/years:5.1f}번 켜짐 · 가장 길게 {max(runs)}일")
    # A 가 켜진 날 중 B 도 켜진 비율
    keys = list(on)
    overlap = {a: {b: round(float(on.loc[on[a], b].mean() * 100), 1) for b in keys} for a in keys}
    print('\n  A가 켜진 날 중 B도 켜진 비율(%)')
    print('  ' + ' ' * 16 + ''.join(f'{k:>9}' for k in keys))
    for a in keys:
        print(f'  {NAMES[a]:<16}' + ''.join(f'{overlap[a][b]:9.0f}' for b in keys))
    # 어느 하나라도 / 몇 개가 같이 켜졌나
    cnt = on.sum(1)
    agree = {int(k): round(float((cnt == k).mean() * 100), 1) for k in range(8)}
    print('\n  7개 중 몇 개가 켜졌나 (날의 %):', agree)
    last = d.iloc[-1]
    snap = {'date': str(last['date'].date()), 'close': float(last['close']), 'rsi': round(float(last['rsi']), 1),
            'pctb': round(float(last['pctb']), 2), 'streak': int(last['streak']), 'ma20': round(float(last['ma20'])),
            'ma60': round(float(last['ma60'])), 'macd': round(float(last['macd'])), 'signal': round(float(last['signal'])),
            'on': {k: bool(on[k].iloc[-1]) for k in keys}}
    print('\n  마지막 날:', snap)
    tail = 180
    recent = {'dates': [str(x.date()) for x in d['date'].iloc[-tail:]], 'close': [float(x) for x in d['close'].iloc[-tail:]],
              'on': {k: [bool(x) for x in on[k].iloc[-tail:]] for k in keys}}
    # 그래프용: 최근 120일 캔들 + 이동평균 · 볼린저 · RSI · MACD
    t = d.iloc[-120:]
    r = lambda v, k=0: [round(float(x), k) if k else round(float(x)) for x in v]
    chart = {'dates': [str(x.date()) for x in t['date']], 'o': r(t['open']), 'h': r(t['high']), 'l': r(t['low']), 'c': r(t['close']),
             'ma20': r(t['ma20']), 'ma60': r(t['ma60']), 'bb_up': r(t['bb_up']), 'bb_lo': r(t['bb_lo']),
             'rsi': r(t['rsi'], 1), 'macd': r(t['macd']), 'signal': r(t['signal']), 'streak': [int(x) for x in t['streak']],
             'pctb': r(t['pctb'], 2)}
    coins = other_coins()
    q = lambda k: [float(np.percentile([r[k] for r in coins if r[k] is not None], p)) for p in (10, 50, 90)]
    print("")
    print(f"다른 코인 {len(coins)}종 (일봉 1년 이상): 10% · 중간 · 90%")
    for k in list(NAMES) + ['rsi_in_above20', 'rsi_in_macd0', 'bb_in_above20', 'above20_in_rsi', 'all7', 'none']:
        print(f"  {NAMES.get(k, k):<18}", ' · '.join(f'{v:5.1f}' for v in q(k)))
    ex = [r for r in coins if r['rsi_in_above20'] is not None and r['rsi_in_above20'] < 100]
    print(f"  RSI 70 초과인데 종가가 20일 평균 아래였던 날이 있는 코인: {len(ex)}종", [(r['coin'], r['rsi_in_above20']) for r in ex[:10]])
    out = {'coins': coins, 'chart': chart, 'from': str(d['date'].iloc[0].date()), 'to': snap['date'], 'days': n, 'stats': stats, 'overlap': overlap,
           'agree': agree, 'snap': snap, 'recent': recent, 'names': NAMES}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
    print('\n저장:', OUT)


if __name__ == '__main__':
    main()
