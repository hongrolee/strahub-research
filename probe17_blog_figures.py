"""probe17 — 블로그 1~9편에 새로 넣은 그림의 자료를 한곳에서 만든다

새로 검증한 것은 없다. 앞선 probe 들이 쓴 자료를 그대로 읽어, 그림으로 그릴 구간만 잘라 낸다.
글(app.strahub.com/blog)의 그림은 이 파일이 낸 outputs/probe17_blog_figures.json 을 옮겨 적은 것이다.

  p1_sig   1편  비트코인 일봉 90일 (2024-01-20 ~ 2024-04-18) + 그날 켜진 조건 셋 (RSI 70 초과 · %b 1 초과 · 3일 연속 상승)
  btc9     1·2편  9년 종가(주 1번) + RSI 70 초과였던 날 전부 (2017-09-25 ~ 2026-09-21)
  p2_rsi   2편  비트코인 일봉 120일 (2023-09-20 ~ 2024-01-17) + RSI(14)
  p2_thr   2편  RSI 기준값 55~80 별 다음날 오른 비율 (글의 표와 같은 계산)
  p3       3편  진짜 종가 vs probe7 과 같은 방법(20일 덩어리 섞기, 씨앗 7)으로 만든 가짜 종가 하나 (주 1번)
  p4       4편  probe8 과 같은 표(740행)에서 비트코인과 이더리움 · 트론의 하루 등락 (%)
  p5       5편  EDGE 급등 5분봉, FLOCK 급등 15분봉 (probe9 의 1분봉을 묶음; FLOCK 은 판 때가 캔들 경계에 오게)
  p6       6편  ADA · 2Z 박스 거래 1시간봉 (probe10 의 15분봉을 묶음)
  p7       7편  비트코인 종가와 200일 평균 (2023-01-01 ~ 2026-09-28, probe11 의 일봉)
  p9       9편  HBAR 9/28~9/30 1시간봉 (data/hbar_1m_20260930.json 을 묶음)

데이터: data/ohlcv_data_kst.db · data/trend_cache · data/spike_cache · data/box_cache · data/hbar_1m_20260930.json (모두 앞 probe 들이 받아 둔 것)
실행:  python probe17_blog_figures.py
결과:  콘솔에 그림 설명에 쓴 사실 + outputs/probe17_blog_figures.json
"""
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
DB = DATA / 'ohlcv_data_kst.db'
OUT = ROOT / 'outputs' / 'probe17_blog_figures.json'
KST = 9 * 3600
END = '2026-09-21'          # 1~3편과 같은 끝날


def r0(v):
    return [int(round(float(x))) for x in v]


def rk(v, k):
    return [round(float(x), k) for x in v]


def btc_daily():
    con = sqlite3.connect('file:' + str(DB) + '?mode=ro', uri=True)
    d = pd.read_sql_query("SELECT timestamp, open_price, high_price, low_price, close_price FROM coin_KRW_BTC "
                          "WHERE interval_type='day' ORDER BY timestamp", con)
    con.close()
    d['date'] = pd.to_datetime(d['timestamp'] - KST, unit='s', utc=True).dt.tz_localize(None).dt.normalize()
    d = d.drop_duplicates('date').sort_values('date')
    d = d[(d['date'] <= END) & (d['close_price'] > 0)].reset_index(drop=True)
    d = d.rename(columns={'open_price': 'open', 'high_price': 'high', 'low_price': 'low', 'close_price': 'close'})
    c = d['close']
    delta = c.diff()                                            # RSI(14), Wilder — probe1 과 같음
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d['rsi'] = 100 - 100 / (1 + gain / loss)
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    d['pctb'] = (c - (ma20 - 2 * sd20)) / (4 * sd20)
    r1 = c.pct_change()
    d['up'] = (r1 > 0).astype(int).groupby((r1 <= 0).cumsum()).cumsum()
    d['fwd'] = c.shift(-1) / c - 1
    return d


def candles(t, unit=1e4):
    """캔들 dict — 비트코인은 만 원 단위 정수로"""
    return {'dates': [str(x.date()) for x in t['date']], 'o': r0(t['open'] / unit), 'h': r0(t['high'] / unit),
            'l': r0(t['low'] / unit), 'c': r0(t['close'] / unit)}


def part_p1(d, out):
    t = d[(d['date'] >= '2024-01-20') & (d['date'] <= '2024-04-18')]
    ch = candles(t)
    ch['rsi70'] = [int(x > 70) for x in t['rsi']]
    ch['bb'] = [int(x > 1) for x in t['pctb']]
    ch['s3'] = [int(x >= 3) for x in t['up']]
    out['p1_sig'] = ch
    print('[p1_sig] %s ~ %s, %d일' % (ch['dates'][0], ch['dates'][-1], len(t)))
    print('   켜진 날  RSI %d · %%b %d · 연속 %d · 하나라도 %d · 셋 다 %d' % (
        sum(ch['rsi70']), sum(ch['bb']), sum(ch['s3']),
        sum(1 for a, b, c in zip(ch['rsi70'], ch['bb'], ch['s3']) if a or b or c),
        sum(1 for a, b, c in zip(ch['rsi70'], ch['bb'], ch['s3']) if a and b and c)))
    print('   종가 %s만 → %s만 (최저 %s만, 최고 %s만)' % (ch['c'][0], ch['c'][-1], min(ch['l']), max(ch['h'])))
    on = [i for i in range(len(t)) if ch['rsi70'][i] or ch['bb'][i] or ch['s3'][i]]
    runs, cur = [], [on[0]]
    for i in on[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            runs.append(cur); cur = [i]
    runs.append(cur)
    for rn in runs:
        a, b = rn[0], rn[-1]
        print('   켜진 묶음 %s ~ %s (%d일)  그 전 10일 종가 변화 %+.1f%%' % (
            ch['dates'][a], ch['dates'][b], len(rn), (t['close'].iloc[a] / t['close'].iloc[max(a - 10, 0)] - 1) * 100))
    up = [i for i in range(len(t)) if ch['c'][i] >= ch['o'][i]]
    onup = sum(1 for i in on if i in up)
    print('   켜진 날 중 그날 양봉 %d / %d' % (onup, len(on)))


def part_btc9(d, out):
    start = d['date'].iloc[0]
    wk = d.iloc[::7]
    if wk.index[-1] != d.index[-1]:
        wk = pd.concat([wk, d.iloc[[-1]]])
    sig = d[d['rsi'] > 70]
    out['btc9'] = {'start': str(start.date()), 'end': str(d['date'].iloc[-1].date()),
                   'line': [[int((a - start).days), int(round(b / 1e4))] for a, b in zip(wk['date'], wk['close'])],
                   'sig': [[int((a - start).days), int(round(b / 1e4))] for a, b in zip(sig['date'], sig['close'])]}
    print('[btc9] RSI 70 초과 %d일, 해마다:' % len(sig), sig.groupby(sig['date'].dt.year).size().to_dict())
    yr = d.groupby(d['date'].dt.year)['close'].agg(['first', 'last'])
    print('   해마다 종가 변화(%):', ((yr['last'] / yr['first'] - 1) * 100).round(0).to_dict())


def part_p2(d, out):
    t = d[(d['date'] >= '2023-09-20') & (d['date'] <= '2024-01-17')]
    ch = candles(t)
    ch['rsi'] = rk(t['rsi'], 1)
    out['p2_rsi'] = ch
    print('[p2_rsi] %s ~ %s, %d일, RSI>70 %d일' % (ch['dates'][0], ch['dates'][-1], len(t), sum(x > 70 for x in ch['rsi'])))
    on = [i for i, x in enumerate(ch['rsi']) if x > 70]
    runs, cur = [], [on[0]]
    for i in on[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            runs.append(cur); cur = [i]
    runs.append(cur)
    for rn in runs:
        a, b = rn[0], rn[-1]
        nxt = [(t['fwd'].iloc[i] > 0) for i in rn]
        after = t['close'].iloc[min(b + 5, len(t) - 1)] / t['close'].iloc[b] - 1
        print('   RSI>70 %s ~ %s (%d일), 들어서기 전 14일 %+.1f%%, 다음날 오른 날 %d/%d, 끝난 뒤 5일 %+.1f%%' % (
            ch['dates'][a], ch['dates'][b], len(rn), (t['close'].iloc[a] / t['close'].iloc[max(a - 14, 0)] - 1) * 100,
            sum(nxt), len(nxt), after * 100))
    print('   종가 %s만 → %s만' % (ch['c'][0], ch['c'][-1]))
    v = d.dropna(subset=['fwd'])
    base = float((v['fwd'] > 0).mean() * 100)
    rows = []
    for th in (55, 60, 65, 70, 75, 80):
        s = v[v['rsi'] > th]
        rows.append([th, len(s), round(float((s['fwd'] > 0).mean() * 100), 1)])
    out['p2_thr'] = {'base': round(base, 1), 'rows': rows}
    print('[p2_thr] 기준선 %.1f%%' % base, rows)


def part_p3(d, out):
    """probe7_surrogate.py 와 같은 방법: 전날 종가 대비 OHLC 비율을 20일 덩어리로 섞어 다시 이음 (씨앗 7, 첫 번째)"""
    rng = np.random.default_rng(7)
    pc = d['close'].shift(1)
    rat = np.column_stack([(d['open'] / pc), (d['high'] / pc), (d['low'] / pc), (d['close'] / pc)])[1:]
    nr, block = len(rat), 20
    nb = int(np.ceil(nr / block))
    st = rng.integers(0, nr - block, nb)
    r = np.concatenate([rat[s:s + block] for s in st])[:nr]
    start = float(d['close'].iloc[0])
    c = start * np.cumprod(r[:, 3])
    prev = np.concatenate([[start], c[:-1]])
    fake = pd.DataFrame({'date': d['date'].iloc[1:].values, 'open': prev * r[:, 0], 'high': prev * r[:, 1],
                         'low': prev * r[:, 2], 'close': c})
    s0 = d['date'].iloc[0]                       # 둘 다 첫날 종가에서 출발
    fc = np.concatenate([[start], c])
    idx = list(range(0, len(d), 7))
    if idx[-1] != len(d) - 1:
        idx.append(len(d) - 1)
    out['p3'] = {
        'start': str(s0.date()), 'end': str(d['date'].iloc[-1].date()),
        'days': [int((d['date'].iloc[i] - s0).days) for i in idx],
        'real': [int(round(d['close'].iloc[i] / 1e4)) for i in idx],
        'fake': [int(round(fc[i] / 1e4)) for i in idx],
    }
    real = d.iloc[1:].reset_index(drop=True)
    print('[p3] 시작 %.0f만, 진짜 끝 %.0f만, 가짜 끝 %.0f만, 가짜 최고 %.0f만 (%s) 최저 %.0f만' % (
        start / 1e4, real['close'].iloc[-1] / 1e4, c[-1] / 1e4, c.max() / 1e4, fake['date'].iloc[int(c.argmax())].date(), c.min() / 1e4))
    rr, rf = real['close'].pct_change().dropna(), pd.Series(c).pct_change().dropna()
    print('   하루 등락 표준편차 진짜 %.2f%% / 가짜 %.2f%%, 하루 최대 상승 %.1f%% / %.1f%%, 최대 하락 %.1f%% / %.1f%%' % (
        rr.std() * 100, rf.std() * 100, rr.max() * 100, rf.max() * 100, rr.min() * 100, rf.min() * 100))
    for nm, t in (('진짜', real.iloc[-60:]), ('가짜', fake.iloc[-60:])):
        print('   %s 마지막 60일: 양봉 %d, 몸통 평균 %.2f%%, 꼬리포함 폭 평균 %.2f%%' % (
            nm, int((t['close'] >= t['open']).sum()), float((abs(t['close'] - t['open']) / t['open']).mean() * 100),
            float(((t['high'] - t['low']) / t['open']).mean() * 100)))


def part_p4(out):
    """probe8_btc_comove.py 와 같은 표: 2024-09-01 ~ 2026-09-23, 95% 이상 거래된 코인, 무거래일 제외, 결측 없는 행"""
    con = sqlite3.connect('file:' + str(DB) + '?mode=ro', uri=True)
    tabs = sorted(t for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'coin_KRW_%'")
                  if re.fullmatch(r"coin_KRW_[A-Z0-9]{2,12}", t))
    close, value = {}, {}
    for t in tabs:
        df = pd.read_sql_query("SELECT timestamp, close_price, value_krw FROM %s WHERE interval_type='day'" % t, con)
        if df.empty:
            continue
        df['d'] = pd.to_datetime(df['timestamp'] - KST, unit='s', utc=True).dt.tz_localize(None).dt.normalize()
        df = df.drop_duplicates('d').set_index('d').sort_index()
        tk = t.replace('coin_KRW_', '')
        close[tk] = df['close_price'].where(df['close_price'] > 0)
        value[tk] = df['value_krw']
    con.close()
    C, V = pd.DataFrame(close), pd.DataFrame(value)
    s, e = pd.Timestamp('2024-09-01'), pd.Timestamp('2026-09-23')
    C, V = C[(C.index >= s) & (C.index <= e)], V[(V.index >= s) & (V.index <= e)]
    cal = C.index[C['BTC'].notna()]
    C, V = C.loc[cal], V.reindex(cal)
    cov = C.notna().sum() / len(cal)
    keep = sorted(c for c in C.columns if cov[c] >= 0.95)
    R = np.log(C[keep]).diff().mask(V[keep].fillna(0) <= 0)
    M = R.dropna()
    res = {'rows': len(M)}
    for tk in ('ETH', 'TRX'):
        cr = float(np.corrcoef(M['BTC'], M[tk])[0, 1])
        pts = [[round(a * 100, 1), round(b * 100, 1)] for a, b in zip(M['BTC'], M[tk])]
        order = M['BTC'].sort_values()
        hi37, lo37 = order.index[-37:], order.index[:37]          # 4편 7절의 «위아래 5%» 74일과 같은 날들
        res[tk] = {'r': round(cr, 3), 'up37': int((M.loc[hi37, tk] > 0).sum()), 'dn37': int((M.loc[lo37, tk] < 0).sum()), 'pts': pts}
        print('[p4] %s  비트코인이 가장 많이 오른 37일 중 같이 오른 날 %d, 가장 많이 내린 37일 중 같이 내린 날 %d' % (tk, res[tk]['up37'], res[tk]['dn37']))
        same = float(((M['BTC'] > 0) == (M[tk] > 0)).mean() * 100)
        big = M['BTC'] <= M['BTC'].quantile(0.05)
        print('[p4] %s  상관 %.3f, %d행, 같은 방향인 날 %.0f%%, 비트코인이 가장 많이 내린 5%% 날(%d일) 중 같이 내린 날 %d일, 등락 범위 %.1f ~ %.1f%%' % (
            tk, cr, len(M), same, int(big.sum()), int((M.loc[big, tk] < 0).sum()), M[tk].min() * 100, M[tk].max() * 100))
    print('   비트코인 등락 범위 %.1f ~ %.1f%%' % (M['BTC'].min() * 100, M['BTC'].max() * 100))
    out['p4'] = res


def m1(coin):
    f = next((DATA / 'spike_cache').glob('KRW-%s_90d_*.json' % coin))
    df = pd.DataFrame(json.loads(f.read_text(encoding='utf-8')))
    df['t'] = pd.to_datetime(df['t'])
    return df.set_index('t')


def bucket(df, t0, k0, k1, step):
    """k(분)마다 (k-step, k] 의 1분봉을 묶은 캔들 — 끝 시각의 종가가 글의 선 그래프 값과 같다"""
    ks, o, h, l, c = [], [], [], [], []
    for k in range(k0, k1 + 1, step):
        a, b = t0 + pd.Timedelta(minutes=k - step + 1), t0 + pd.Timedelta(minutes=k)
        w = df[(df.index >= a) & (df.index <= b)]
        if w.empty:
            last = float(df[df.index <= b]['c'].iloc[-1])
            ks.append(k); o.append(last); h.append(last); l.append(last); c.append(last)
            continue
        ks.append(k); o.append(float(w['o'].iloc[0])); h.append(float(w['h'].max())); l.append(float(w['l'].min())); c.append(float(w['c'].iloc[-1]))
    return {'k': ks, 'o': o, 'h': h, 'l': l, 'c': c}


def part_p5(out):
    cases = {'EDGE': ('2026-09-19T05:50:00', -120, 195, 5, 4, 75),       # sig_t, 처음, 끝, 캔들 분, 판 k, 다시 산 k (글의 _SR 와 같음)
             'FLOCK': ('2026-09-04T14:23:00', -115, 1550, 15, 5, 1445)}   # FLOCK 은 판 때(k=5)가 캔들 경계에 오게 끊음
    res = {}
    for m, (sig, k0, k1, step, sk, rk_) in cases.items():
        t0 = pd.Timestamp(sig)
        ch = bucket(m1(m), t0, k0, k1, step)
        res[m] = dict(ch, step=step)
        hi_i = int(np.argmax(ch['h']))
        after = [i for i, k in enumerate(ch['k']) if k - step >= sk]
        lo_i = min(after, key=lambda i: ch['l'][i])
        print('[p5] %s %d분봉 %d개, 가장 높은 고가 %g (k=%d), 판 뒤 가장 낮은 저가 %g (k=%d)' % (
            m, step, len(ch['k']), ch['h'][hi_i], ch['k'][hi_i], ch['l'][lo_i], ch['k'][lo_i]))
        big = sorted(range(len(ch['k'])), key=lambda i: ch['c'][i] - ch['o'][i])
        print('   가장 큰 음봉 k=%d (%g→%g), 가장 큰 양봉 k=%d (%g→%g)' % (
            ch['k'][big[0]], ch['o'][big[0]], ch['c'][big[0]], ch['k'][big[-1]], ch['o'][big[-1]], ch['c'][big[-1]]))
        rb = [i for i, k in enumerate(ch['k']) if k - step < rk_ <= k][0]
        print('   다시 산 캔들 k=%d  o/h/l/c = %g/%g/%g/%g' % (ch['k'][rb], ch['o'][rb], ch['h'][rb], ch['l'][rb], ch['c'][rb]))
    out['p5'] = res


def hourly15(coin, start, n):
    f = next((DATA / 'box_cache').glob('KRW-%s_15m_365d_*.json' % coin))
    df = pd.DataFrame(json.loads(f.read_text(encoding='utf-8')))
    df['t'] = pd.to_datetime(df['t'])
    df = df.set_index('t')
    st = pd.Timestamp(start)
    o, h, l, c = [], [], [], []
    for k in range(n):
        w = df[(df.index >= st + pd.Timedelta(hours=k)) & (df.index < st + pd.Timedelta(hours=k + 1))]
        o.append(float(w['o'].iloc[0])); h.append(float(w['h'].max())); l.append(float(w['l'].min())); c.append(float(w['c'].iloc[-1]))
    return {'o': o, 'h': h, 'l': l, 'c': c}


def part_p6(out):
    cases = {'ADA': ('2026-08-22T02:00:00', 52, 293.0, 321.0, 6, 30), '2Z': ('2026-09-26T05:00:00', 45, 91.0, 103.0, 6, 30)}
    res = {}
    for m, (start, n, lo, hi, bf, bk) in cases.items():
        ch = hourly15(m, start, n)
        res[m] = ch
        bl, bh = min(ch['l'][bf:bk]), max(ch['h'][bf:bk])
        print('[p6] %s 1시간봉 %d개, 박스 기간(k %d~%d) 저가 최저 %g · 고가 최고 %g (글의 박스 %g~%g)' % (m, n, bf, bk - 1, bl, bh, lo, hi))
        print('   전체 저가 최저 %g (k=%d), 고가 최고 %g (k=%d); 산 뒤 저가 최저 %g' % (
            min(ch['l']), int(np.argmin(ch['l'])), max(ch['h']), int(np.argmax(ch['h'])), min(ch['l'][bk:])))
    out['p6'] = res


def part_p7(out):
    f = next((DATA / 'trend_cache').glob('KRW-BTC_1d_2000d_*.json'))
    raw = json.loads(f.read_text(encoding='utf-8'))
    d = pd.DataFrame({'date': pd.to_datetime([r['d'] for r in raw]), 'close': [r['c'] for r in raw]})
    d = d[d['date'] <= '2026-09-28'].reset_index(drop=True)
    d['ma200'] = d['close'].rolling(200).mean()
    t = d[d['date'] >= '2023-01-01'].reset_index(drop=True)
    above = t['close'] > t['ma200']
    out['p7'] = {'start': str(t['date'].iloc[0].date()), 'end': str(t['date'].iloc[-1].date()),
                 'c': r0(t['close'] / 1e4), 'ma': r0(t['ma200'] / 1e4)}
    print('[p7] %s ~ %s, %d일, 종가 > 200일 평균인 날 %.1f%%' % (out['p7']['start'], out['p7']['end'], len(t), above.mean() * 100))
    runs, cur, st = [], None, None
    for i, a in enumerate(above):
        if cur is None or a != cur:
            if cur is not None:
                runs.append((cur, st, i - 1))
            cur, st = a, i
    runs.append((cur, st, len(t) - 1))
    for a, s_, e_ in runs:
        if e_ - s_ >= 20:
            print('   %s %s ~ %s (%d일)  종가 %.0f만 → %.0f만' % ('위' if a else '아래', t['date'].iloc[s_].date(), t['date'].iloc[e_].date(),
                                                         e_ - s_ + 1, t['close'].iloc[s_] / 1e4, t['close'].iloc[e_] / 1e4))
    print('   위↔아래 바뀐 횟수 %d' % (len(runs) - 1))
    for y in (2023, 2024, 2025, 2026):
        m = t['date'].dt.year == y
        print('   %d년 위인 날 %.0f%%' % (y, above[m].mean() * 100))
    print('   최고 종가 %.0f만 (%s)' % (t['close'].max() / 1e4, t['date'].iloc[int(t['close'].idxmax())].date()))


def part_p9(out):
    raw = json.loads((DATA / 'hbar_1m_20260930.json').read_text(encoding='utf-8'))
    df = pd.DataFrame(raw['b'], columns=['o', 'h', 'l', 'c'])
    df['t'] = pd.to_datetime(raw['t'])
    df = df.set_index('t')
    st, en = pd.Timestamp('2026-09-28T12:00'), pd.Timestamp('2026-09-30T06:53')
    df = df[(df.index >= st) & (df.index <= en)]
    g = df.groupby(df.index.floor('h'))
    t = pd.DataFrame({'o': g['o'].first(), 'h': g['h'].max(), 'l': g['l'].min(), 'c': g['c'].last()})
    out['p9'] = {'start': str(t.index[0]), 'o': rk(t['o'], 2), 'h': rk(t['h'], 2), 'l': rk(t['l'], 2), 'c': rk(t['c'], 2)}
    print('[p9] HBAR 1시간봉 %d개 (%s ~ %s)' % (len(t), t.index[0], t.index[-1]))
    print('   가장 높은 고가 %g (%s), 마지막 종가 %g' % (t['h'].max(), t['h'].idxmax(), t['c'].iloc[-1]))
    big = (t['c'] - t['o']).sort_values()
    print('   가장 큰 양봉 %s (%g→%g), 가장 큰 음봉 %s (%g→%g)' % (
        big.index[-1], t.loc[big.index[-1], 'o'], t.loc[big.index[-1], 'c'], big.index[0], t.loc[big.index[0], 'o'], t.loc[big.index[0], 'c']))
    for a, b in (('2026-09-28 17:00', '2026-09-29 03:00'), ('2026-09-29 04:00', '2026-09-30 06:00')):
        w = t[(t.index >= a) & (t.index <= b)]
        print('   %s ~ %s  양봉 %d / %d' % (a, b, int((w['c'] >= w['o']).sum()), len(w)))


def main():
    out = {}
    d = btc_daily()
    part_p1(d, out)
    part_btc9(d, out)
    part_p2(d, out)
    part_p3(d, out)
    part_p4(out)
    part_p5(out)
    part_p6(out)
    part_p7(out)
    part_p9(out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print('\n저장:', OUT, '(%d KB)' % (OUT.stat().st_size // 1024))


if __name__ == '__main__':
    main()
