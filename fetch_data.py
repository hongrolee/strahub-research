# -*- coding: utf-8 -*-
"""
업비트 공개 API에서 일봉을 받아 검증용 SQLite 를 만든다.

인증키가 필요 없다. 업비트의 시세 조회 API 는 누구나 호출할 수 있다.

    python fetch_data.py                # KRW-BTC 만 (약 17번 호출, 10초)
    python fetch_data.py --all-coins    # 원화 마켓 전체 (약 12분, probe4 에 필요)
    python fetch_data.py --coins ETH XRP DOGE

만들어지는 파일: data/ohlcv_data_kst.db  (BTC 만 받으면 몇 MB)

저장 형식은 StraHub 본 서비스의 DB 와 같다. 그래서 probe 스크립트를 고치지 않고 쓸 수 있다.
  테이블   coin_KRW_BTC
  컬럼     interval_type, timestamp, open_price, high_price, low_price, close_price, volume
  timestamp  KST 날짜를 UTC 자정으로 본 epoch + 9시간
             (읽을 때 9시간을 빼면 그 날짜가 나온다 — probe 들이 그렇게 읽는다)
"""
import argparse
import calendar
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

API = "https://api.upbit.com/v1"
DB_PATH = "data/ohlcv_data_kst.db"
KST = timezone(timedelta(hours=9))
UA = {"User-Agent": "strahub-research/1.0 (+https://app.strahub.com/blog)"}
START = datetime(2017, 9, 1, tzinfo=timezone.utc)   # 업비트 원화 마켓 개장 무렵
PAGE = 200                                          # 업비트 일봉 1회 최대
SLEEP = 0.15                                        # 초당 10회 제한 여유 있게


def get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:                        # 호출 제한 — 기다렸다 재시도
                time.sleep(1.5 * (i + 1))
                continue
            raise
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1.0 * (i + 1))
    return []


def kst_date_to_ts(s: str) -> int:
    """'2026-09-22T00:00:00' (KST 기준 일봉 라벨) -> 저장용 timestamp"""
    d = datetime.strptime(s[:10], "%Y-%m-%d")
    utc_midnight = calendar.timegm((d.year, d.month, d.day, 0, 0, 0, 0, 0, 0))
    return utc_midnight + 9 * 3600


def ensure_table(con, table):
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            interval_type TEXT NOT NULL,
            timestamp     INTEGER NOT NULL,
            open_price    REAL,
            high_price    REAL,
            low_price     REAL,
            close_price   REAL,
            volume        REAL,
            value_krw     REAL,
            created_at    TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at    TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(interval_type, timestamp)
        )""")
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table} ON {table}(interval_type, timestamp DESC)")


def fetch_market(con, market: str) -> int:
    """market 예: 'KRW-BTC'. 오래된 쪽으로 거슬러 올라가며 전부 받는다."""
    table = "coin_" + market.replace("-", "_")
    ensure_table(con, table)

    to = datetime.now(timezone.utc)
    rows, seen = [], set()
    while True:
        url = f"{API}/candles/days?market={market}&count={PAGE}&to={to.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        page = get(url)
        if not page:
            break
        oldest = None
        for c in page:
            ts = kst_date_to_ts(c["candle_date_time_kst"])
            if ts in seen:
                continue
            seen.add(ts)
            rows.append((
                "day", ts,
                c["opening_price"], c["high_price"], c["low_price"], c["trade_price"],
                c["candle_acc_trade_volume"], c.get("candle_acc_trade_price"),
            ))
            u = datetime.strptime(c["candle_date_time_utc"][:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            oldest = u if oldest is None or u < oldest else oldest
        if oldest is None or oldest <= START or len(page) < PAGE:
            break
        to = oldest - timedelta(seconds=1)
        time.sleep(SLEEP)

    if rows:
        con.executemany(
            f"""INSERT OR REPLACE INTO {table}
                (interval_type, timestamp, open_price, high_price, low_price,
                 close_price, volume, value_krw)
                VALUES (?,?,?,?,?,?,?,?)""", rows)
        con.commit()
    return len(rows)


def krw_markets():
    data = get(f"{API}/market/all?isDetails=false")
    return [m["market"] for m in data if m["market"].startswith("KRW-")]


def main():
    ap = argparse.ArgumentParser(description="업비트 일봉을 받아 검증용 SQLite 생성")
    ap.add_argument("--coins", nargs="*", metavar="SYMBOL",
                    help="받을 코인 심볼 (예: ETH XRP). 생략하면 BTC 만")
    ap.add_argument("--all-coins", action="store_true",
                    help="원화 마켓 전체 (probe4 교차검증에 필요, 약 12분)")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    if args.all_coins:
        markets = krw_markets()
        print(f"원화 마켓 {len(markets)}개를 받습니다. 10분 이상 걸립니다.\n")
    elif args.coins:
        markets = ["KRW-BTC"] + [f"KRW-{c.upper()}" for c in args.coins if c.upper() != "BTC"]
    else:
        markets = ["KRW-BTC"]

    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.db)), exist_ok=True)
    con = sqlite3.connect(args.db)
    con.execute("PRAGMA journal_mode=WAL")

    total = 0
    for i, m in enumerate(markets, 1):
        try:
            n = fetch_market(con, m)
            total += n
            print(f"[{i:3d}/{len(markets)}] {m:<12} {n:>6}일")
        except Exception as e:
            print(f"[{i:3d}/{len(markets)}] {m:<12} 실패: {type(e).__name__} {e}")
        time.sleep(SLEEP)

    cur = con.execute("SELECT MIN(timestamp), MAX(timestamp), COUNT(*) "
                      "FROM coin_KRW_BTC WHERE interval_type='day'").fetchone()
    con.close()

    def lab(ts):
        return datetime.fromtimestamp(ts - 9 * 3600, tz=timezone.utc).strftime("%Y-%m-%d")

    print(f"\n완료: {args.db}")
    print(f"  전체 저장 행수 : {total:,}")
    if cur and cur[2]:
        print(f"  KRW-BTC 일봉   : {cur[2]:,}일  ({lab(cur[0])} ~ {lab(cur[1])})")
    print("\n다음 단계:  python probe1_baseline.py")
    print("주의: 오늘 날짜 캔들은 아직 마감 전이라 probe 스크립트가 자동으로 제외합니다.")


if __name__ == "__main__":
    sys.exit(main())
