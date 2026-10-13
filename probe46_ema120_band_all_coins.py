"""
probe46: '120일 지수이동평균(EMA) 위에서 들고 · 아래 2% 넘게 내려가면 팖'을 (probe45 의 50일 단순 평균과 함께) 비트코인 말고 다른 코인마다 — 그 코인을 그냥 들고 있기와 비교
- 자료: data/ohlcv_data_kst.db 일봉 (지금 업비트 원화 코인, 상장폐지 코인 없음)
- 코인마다 2019-01-01 또는 상장 60일 뒤부터 끝까지 · 100만 원 · 사고팔 때마다 0.15% · 어제 종가로 정하고 오늘 종가에 사고팖
- 함께: 대칭 100일 · 대칭 50일 (같은 코인)
- 해마다: 그해 1월 1일 100만 원으로 새로 (그해 전부터 있던 코인만)
결과: outputs/probe46_ema120_band_all_coins.json
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FEE = 0.0015
STABLE = {'USDT', 'USDC', 'USDE', 'USD1', 'USDG', 'USDS', 'PYUSD', 'RLUSD', 'FDUSD', 'TUSD', 'DAI', 'EURC', 'JPYC'}
RULES = {'EMA120 · 아래 2%': ('ema', 120, 0, 2), 'EMA50 · 아래 2%': ('ema', 50, 0, 2), 'SMA50 · 아래 2%': ('sma', 50, 0, 2), 'SMA120 · 아래 2%': ('sma', 120, 0, 2)}


def load():
    con = sqlite3.connect(f"file:{HERE / 'data' / 'ohlcv_data_kst.db'}?mode=ro", uri=True)
    out = {}
    for (t,) in con.execute("select name from sqlite_master where type='table'"):
        if not t.startswith('coin_KRW_') or t[9:] in STABLE:
            continue
        try:
            rows = con.execute(f'select timestamp, close_price from "{t}" where interval_type=\'day\' order by timestamp').fetchall()
        except Exception:
            continue
        rows = [(datetime.fromtimestamp(ts).strftime('%Y-%m-%d'), c) for ts, c in rows if c]
        if len(rows) >= 200:
            out[t[9:]] = rows
    return out


def run(C, kind, n, b, s, t0, t1):
    if kind == 'ema':                                  # 전체 기록으로 이어 계산 (차트의 EMA 와 같게), 앞 n봉은 쓰지 않음
        ma = np.empty(len(C))
        ma[0] = C[0]
        k = 2 / (n + 1)
        for i in range(1, len(C)):
            ma[i] = C[i] * k + ma[i - 1] * (1 - k)
        ma[:n - 1] = np.nan
    else:
        ma = np.convolve(C, np.ones(n) / n, mode='full')[:len(C)]
        ma[:n - 1] = np.nan
    cash, qty, peak, mdd, trades = 1.0, 0.0, 1.0, 0.0, 0
    for t in range(t0, t1 + 1):
        if np.isnan(ma[t - 1]):
            v = cash + qty * C[t]
        else:
            r = C[t - 1] / ma[t - 1] - 1
            if qty == 0 and r > b / 100:
                qty, cash, trades = cash * (1 - FEE) / C[t], 0.0, trades + 1
            elif qty > 0 and r < -s / 100:
                cash, qty, trades = qty * C[t] * (1 - FEE), 0.0, trades + 1
            v = cash + qty * C[t]
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    return cash + qty * C[t1] * (1 - FEE), mdd * 100, trades


def hold(C, t0, t1):
    seg = C[t0:t1 + 1] / C[t0]
    return seg[-1] * (1 - FEE) ** 2, float((seg / np.maximum.accumulate(seg) - 1).min() * 100)


def main():
    data = load()
    end = max(r[-1][0] for r in data.values())
    res = {}
    for sym, rows in data.items():
        days = [d for d, _ in rows]
        C = np.array([c for _, c in rows], dtype=float)
        t0 = max(next((k for k, d in enumerate(days) if d >= '2019-01-01'), len(days)), 130)
        if t0 >= len(days) - 60:
            continue
        t1 = len(days) - 1
        h, hm = hold(C, t0, t1)
        r = {'from': days[t0], 'days': t1 - t0, 'hold': h, 'hold_mdd': hm}
        for name, (kind, n, b, s) in RULES.items():
            f, m, tr = run(C, kind, n, b, s, t0, t1)
            r[name] = {'final': f, 'mdd': m, 'trades': tr}
        # 해마다
        yrs = {}
        for y in range(2019, int(end[:4]) + 1):
            a = next((k for k, d in enumerate(days) if d >= f'{y}-01-01'), None)
            z = max((k for k, d in enumerate(days) if d <= f'{y}-12-31'), default=None)
            if a is None or z is None or a < 130 or z - a < 60 or days[a][:4] != str(y):
                continue
            yh, _ = hold(C, a, z)
            yrs[y] = {'hold': yh, **{name: run(C, kind, n, b, s, a, z)[0] for name, (kind, n, b, s) in RULES.items()}}
        r['years'] = yrs
        res[sym] = r
    coins = sorted(res)
    print(f"코인 {len(coins)}개 (일봉 · 2019-01 또는 상장 110일 뒤부터 ~ {end}) · 코인마다 그 코인을 그냥 들고 있기와 비교")
    for name in RULES:
        win = [c for c in coins if res[c][name]['final'] > res[c]['hold']]
        ratio = np.array([res[c][name]['final'] / res[c]['hold'] for c in coins])
        dd = np.array([res[c][name]['mdd'] - res[c]['hold_mdd'] for c in coins])
        both_up = [c for c in coins if res[c]['hold'] > 1]
        lost_upside = np.median([res[c][name]['final'] / res[c]['hold'] for c in both_up]) if both_up else float('nan')
        print(f"  {name:<12} 들고 있기보다 나은 코인 {len(win)}/{len(coins)} ({len(win) / len(coins) * 100:.0f}%) · 결과/들고 있기 가운데 값 {np.median(ratio):.2f}배"
              f" · 가장 크게 줄었을 때 가운데 {np.median([res[c][name]['mdd'] for c in coins]):.0f}% (들고 있기 {np.median([res[c]['hold_mdd'] for c in coins]):.0f}%)"
              f" · 오른 코인({len(both_up)}개)에서는 가운데 {lost_upside:.2f}배 · 거래 가운데 {np.median([res[c][name]['trades'] for c in coins]):.0f}번")
    print('\n큰 코인 몇 개 (최종 금액 배수: 들고 있기 / 50일 · 아래 2% / 100일 대칭 · 가장 크게 줄었을 때)')
    for c in ['BTC', 'ETH', 'XRP', 'SOL', 'DOGE', 'ADA', 'LINK', 'TRX', 'AVAX', 'DOT', 'BCH', 'ETC', 'SAND', 'HBAR', 'SUI']:
        if c in res:
            r = res[c]
            print(f"  {c:<5} {r['from']}~ 들고 {r['hold']:7.2f}배 ({r['hold_mdd']:.0f}%) · EMA120·2% {r['EMA120 · 아래 2%']['final']:7.2f}배 ({r['EMA120 · 아래 2%']['mdd']:.0f}%, {r['EMA120 · 아래 2%']['trades']}번) · SMA50·2% {r['SMA50 · 아래 2%']['final']:7.2f}배 ({r['SMA50 · 아래 2%']['mdd']:.0f}%, {r['SMA50 · 아래 2%']['trades']}번)")
    print('\n해마다 (그해 있던 코인들, 들고 있기보다 나은 코인 비율 · 가운데 결과)')
    for y in range(2019, int(end[:4]) + 1):
        cs = [c for c in coins if y in res[c]['years']]
        if not cs:
            continue
        hy = np.median([res[c]['years'][y]['hold'] for c in cs])
        line = f"  {y} ({len(cs)}개, 들고 있기 가운데 {hy:.2f}배)"
        for name in RULES:
            w = sum(1 for c in cs if res[c]['years'][y][name] > res[c]['years'][y]['hold'])
            line += f" · {name} {w / len(cs) * 100:.0f}% ({np.median([res[c]['years'][y][name] for c in cs]):.2f}배)"
        print(line)
    (HERE / 'outputs' / 'probe46_ema120_band_all_coins.json').write_text(json.dumps(res, ensure_ascii=False, default=float), encoding='utf-8')


if __name__ == '__main__':
    main()
