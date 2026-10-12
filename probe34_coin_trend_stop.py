"""
probe34: 코인마다 자기 추세(Supertrend)에 따라 손절을 바꾸면? — probe29 는 '시장 전체' 분위기, 이번은 '그 코인' 추세
- 날마다 어제 종가까지 본 그 코인의 Supertrend(일봉 10·3 / 7·2)가 내림이면 손절 a%, 오름이면 b% (a,b = 10/20, 10/15, 15/20)
- 또 트레이딩뷰에서 흔히 쓰는 방법: Supertrend 선 자체를 손절선으로 (종가가 선 아래로 내려가면 팖, 아래 손절 15 는 그대로)
- 사는 법 · 기간: probe25 (7일마다 급등잦던 / 모든 코인 같은 금액씩, 30 · 90일, 익절 50, 비용 한쪽 0.15%)
- 대조: 날마다 같은 비율로 아무 날을 '내림'으로 정해 손절을 바꿈 (그 코인 추세에 정보가 있는지)
결과: outputs/probe34_coin_trend_stop.json
"""
import json
from pathlib import Path

import numpy as np

from probe25_bot3_settings import load, spike_prone, COST
from probe30_supertrend import supertrend

HERE = Path(__file__).resolve().parent
TAKE = 50
RNG = np.random.default_rng(34)


def sim(e, i, t0, hold, H, L, C, down, lo_s, hi_s, fixed=None, line=None):
    hi = e * (1 + TAKE / 100)
    last = e
    for k in range(1, hold + 1):
        t = t0 + k
        if t >= C.shape[1] or np.isnan(C[i, t]):
            return last
        s = fixed if fixed else (lo_s if down[i, t - 1] else hi_s)
        if L[i, t] <= e * (1 - s / 100):
            return e * (1 - s / 100)
        if H[i, t] >= hi:
            return hi
        last = C[i, t]
        if line is not None and line[i, t - 1] == 1 and line[i, t] == -1:      # 종가가 Supertrend 선 아래로 → 그날 종가에 팖
            return last
    return last


def main():
    data = load()
    mk, days, H, L, C, V, first, binance = data
    sts = {'10·3': supertrend(H, L, C, 10, 3.0), '7·2': supertrend(H, L, C, 7, 2.0)}
    out = {}
    for hold in (30, 90):
        for pick in ('spike', 'all'):
            starts = list(range(30, len(days) - hold, 7))
            mid = starts[len(starts) // 2]
            pos = [(t0, i) for t0 in starts for i in range(len(mk)) if not np.isnan(C[i, t0]) and t0 - first[i] >= 5
                   and (pick == 'all' or spike_prone(i, t0, mk, C, V, first, binance, 0))]

            def score(fn):
                f, b = [], []
                for t0, i in pos:
                    e = C[i, t0]
                    r = (fn(e, i, t0) * (1 - COST)) / (e * (1 + COST)) - 1
                    (f if t0 < mid else b).append(r)
                return {'avg': float(np.mean(f + b) * 100), 'front': float(np.mean(f) * 100), 'back': float(np.mean(b) * 100)}
            rows = [{'rule': f'늘 손절 {s}', **score(lambda e, i, t0, s=s: sim(e, i, t0, hold, H, L, C, None, 0, 0, fixed=s))} for s in (10, 15, 20)]
            for key, D in sts.items():
                down = D == -1
                p = down[:, 30:][D[:, 30:] != 0].mean()
                rnd = RNG.random(D.shape) < p
                for lo_s, hi_s in ((10, 20), (10, 15), (15, 20)):
                    r = {'rule': f'ST {key} 내림 {lo_s} / 오름 {hi_s}', **score(lambda e, i, t0, down=down: sim(e, i, t0, hold, H, L, C, down, lo_s, hi_s))}
                    r['rand'] = score(lambda e, i, t0: sim(e, i, t0, hold, H, L, C, rnd, lo_s, hi_s))
                    rows.append(r)
                rows.append({'rule': f'ST {key} 선 아래로 가면 팖 (+ 손절 15)', **score(lambda e, i, t0, D=D: sim(e, i, t0, hold, H, L, C, None, 0, 0, fixed=15, line=D))})
            out[f'{pick}_{hold}'] = rows
            base = rows[1]
            print(f"\n== {'급등이 잦던 코인' if pick == 'spike' else '모든 코인'} · {hold}일 (앞/뒤 나눔 {days[mid]}) n={len(pos)}")
            for r in rows:
                mark = ' ◎' if r is not base and r['front'] > base['front'] and r['back'] > base['back'] else ''
                line = f"  {r['rule']:<30} 평균 {r['avg']:+6.2f}% | 앞 {r['front']:+6.2f} 뒤 {r['back']:+6.2f}{mark}"
                if 'rand' in r:
                    line += f"   (아무 날 대조 {r['rand']['avg']:+6.2f} → 추세 몫 {r['avg'] - r['rand']['avg']:+5.2f})"
                print(line)
    (HERE / 'outputs' / 'probe34_coin_trend_stop.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
