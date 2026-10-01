"""probe18 데이터: 업비트 KRW 전체 코인(유의 종목 포함, 스테이블 제외)의 최근 90일 1분봉

    python probe18_fetch.py            # 받기 (이어받기 됨 — 끊겨도 다시 실행하면 남은 코인만)
    python probe18_fetch.py status     # 몇 개 받았는지

- 업비트 공개 시세 API만 씀 (키·계정 없음). 초당 8회 이하로 보내고, 429면 기다렸다 다시.
- 기간은 처음 실행할 때 정해 data/spike_cache_all/_window.json 에 적어 두고, 이어받을 때도 같은 기간을 씀.
- 코인 목록과 '받는 날의 유의·주의 표시'는 _markets.json 에 (과거 유의 이력은 업비트가 주지 않음).
- 이미 받아 둔 상위 50개(trading_bot_template/data/spike_cache, 2026-09-27까지)는 재사용하고 뒤쪽만 이어 받음.
- 저장: 코인마다 <market>.npz (t: UTC 분 단위 epoch 초, o/h/l/c: 가격, v: 1분 거래대금 원). 빈 분은 저장하지 않음(분석 때 0으로 채움).
"""
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / 'data' / 'spike_cache_all'
OLD = Path(r'E:\MAYAI\AutoTrading_Flask\trading_bot_template\data\spike_cache')
DAYS = 90
RATE = 8.0                       # 초당 요청 수 (업비트 시세 캔들: 초당 10회 제한)
STABLE = {'KRW-USDT', 'KRW-USDC', 'KRW-USD1', 'KRW-USDS', 'KRW-DAI'}


class Pacer:
    def __init__(self, rate):
        self.gap = 1.0 / rate
        self.last = 0.0
        self.calls = 0

    def wait(self):
        d = self.last + self.gap - time.monotonic()
        if d > 0:
            time.sleep(d)
        self.last = time.monotonic()
        self.calls += 1


S = requests.Session()


def get(pacer, url, params):
    for attempt in range(8):
        pacer.wait()
        try:
            r = S.get(url, params=params, timeout=15)
        except requests.RequestException:
            time.sleep(2 + attempt)
            continue
        if r.status_code == 429:
            time.sleep(1 + attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f'요청이 계속 실패해요: {url} {params}')


def window():
    f = OUT / '_window.json'
    if f.exists():
        w = json.loads(f.read_text(encoding='utf-8'))
        return datetime.fromisoformat(w['start']), datetime.fromisoformat(w['end'])
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=DAYS)
    f.write_text(json.dumps({'start': start.isoformat(), 'end': end.isoformat()}), encoding='utf-8')
    return start, end


def markets(pacer):
    f = OUT / '_markets.json'
    if f.exists():
        return json.loads(f.read_text(encoding='utf-8'))
    allm = get(pacer, 'https://api.upbit.com/v1/market/all', {'isDetails': 'true'})
    krw = [m for m in allm if m['market'].startswith('KRW-') and m['market'] not in STABLE]
    ticks = {}
    names = [m['market'] for m in krw]
    for k in range(0, len(names), 100):
        for t in get(pacer, 'https://api.upbit.com/v1/ticker', {'markets': ','.join(names[k:k + 100])}):
            ticks[t['market']] = t
    rows = []
    for m in krw:
        t = ticks.get(m['market'], {})
        ev = m.get('market_event') or {}
        rows.append({'market': m['market'], 'korean_name': m.get('korean_name'), 'english_name': m.get('english_name'),
                     'warning': m.get('market_warning') == 'CAUTION' or bool(ev.get('warning')),
                     'caution': {k: v for k, v in (ev.get('caution') or {}).items() if v},
                     'price': t.get('trade_price'), 'value_24h': t.get('acc_trade_price_24h')})
    rows.sort(key=lambda r: -(r['value_24h'] or 0))
    f.write_text(json.dumps({'fetched_at': datetime.now(timezone.utc).isoformat(), 'markets': rows},
                            ensure_ascii=False, indent=1), encoding='utf-8')
    return json.loads(f.read_text(encoding='utf-8'))


def _epoch(s):
    return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp())


def old_bars(market, start_ep):
    """상위 50개 예전 캐시 (있으면): {epoch: (o,h,l,c,v)} — 빈 분을 0으로 채운 행(v=0)은 버림"""
    files = sorted(OLD.glob(f'{market}_90d_*.json'))
    if not files:
        return {}
    out = {}
    for b in json.loads(files[-1].read_text(encoding='utf-8')):
        e = _epoch(b['t'])
        if e >= start_ep and b.get('v'):
            out[e] = (b['o'], b['h'], b['l'], b['c'], b['v'])
    return out


def fetch_market(pacer, market, start, end):
    start_ep, end_ep = int(start.timestamp()), int(end.timestamp())
    rows = old_bars(market, start_ep)
    stop_ep = max(rows) if rows else start_ep             # 예전 캐시가 있으면 그 뒤만
    to = end
    while True:
        chunk = get(pacer, 'https://api.upbit.com/v1/candles/minutes/1',
                    {'market': market, 'count': 200, 'to': to.strftime('%Y-%m-%dT%H:%M:%SZ')})
        if not chunk:
            break
        for c in chunk:
            e = _epoch(c['candle_date_time_utc'])
            if start_ep <= e < end_ep:
                rows[e] = (c['opening_price'], c['high_price'], c['low_price'], c['trade_price'], c['candle_acc_trade_price'])
        oldest = _epoch(chunk[-1]['candle_date_time_utc'])
        if oldest <= stop_ep or len(chunk) < 200:
            break
        to = datetime.fromtimestamp(oldest, timezone.utc)
    if not rows:
        return 0
    ts = np.array(sorted(rows), dtype=np.int64)
    arr = np.array([rows[t] for t in ts], dtype=np.float64)
    np.savez_compressed(OUT / f'{market}.npz', t=ts, o=arr[:, 0], h=arr[:, 1], l=arr[:, 2], c=arr[:, 3], v=arr[:, 4])
    return len(ts)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pacer = Pacer(RATE)
    start, end = window()
    mk = markets(pacer)['markets']
    todo = [m['market'] for m in mk if not (OUT / f"{m['market']}.npz").exists()]
    if len(sys.argv) > 1 and sys.argv[1] == 'status':
        print(f'{len(mk) - len(todo)}/{len(mk)} 코인 받음 · 기간 {start:%Y-%m-%d %H:%M} ~ {end:%Y-%m-%d %H:%M} UTC')
        return
    print(f'기간 {start:%Y-%m-%d %H:%M} ~ {end:%Y-%m-%d %H:%M} UTC · 남은 코인 {len(todo)}/{len(mk)}', flush=True)
    t0 = time.time()
    for i, market in enumerate(todo, 1):
        n = fetch_market(pacer, market, start, end)
        el = time.time() - t0
        left = el / i * (len(todo) - i)
        print(f'[{i}/{len(todo)}] {market}: {n:,}봉 · 요청 {pacer.calls:,}번 · 남은 시간 약 {left / 60:.0f}분', flush=True)
    print('끝', flush=True)


if __name__ == '__main__':
    main()
