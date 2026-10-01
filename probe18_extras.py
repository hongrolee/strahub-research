"""probe18 보충 계산 (probe18_spike_traits.py 결과를 읽음)

- 시장 전체 급등(같은 10분 칸에 5개 넘는 코인) / 코인 혼자 급등으로 나눔
- 급등이 끝난 뒤 산 사람 기준(급등 1분의 종가) 30분·4시간 뒤 가격
- 직전 신호 두 개가 겹칠 때 '다음 60분 안 급등' 비율
→ outputs/probe18_extras.json
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

import probe18_spike_traits as P

OUT = P.OUT


def main():
    r = json.loads((OUT / 'probe18_spike_traits.json').read_text(encoding='utf-8'))
    ev = r['events']
    slot = Counter(e['i'] // 10 for e in ev)
    for e in ev:
        e['wide'] = slot[e['i'] // 10] > 5
    alone = [e for e in ev if not e['wide']]
    x = {'wide_events': sum(e['wide'] for e in ev), 'alone_events': len(alone),
         'alone_hour': [sum(e['hour'] == h for e in alone) / len(alone) * 100 for h in range(24)],
         'alone_weekday': [sum(e['weekday'] == d for e in alone) for d in range(7)],
         'alone_weekly': Counter(), 'nine_oclock': {}}
    # 9시 정각 (KST 09:00) 비중
    nine = [e for e in alone if e['hour'] == 9 and P.kst_dt(e['i']).minute == 0]
    x['nine_oclock'] = {'events': len(nine), 'share': len(nine) / len(alone) * 100,
                        'days': len({e['date'] for e in nine}),
                        'hour9_share': sum(e['hour'] == 9 for e in alone) / len(alone) * 100}
    from datetime import datetime, timedelta
    wk = Counter()
    for e in alone:
        d = datetime.strptime(e['date'], '%Y-%m-%d')
        wk[(d - timedelta(days=d.weekday())).strftime('%m-%d')] += 1
    x['alone_weekly'] = [{'week': k, 'events': wk[k]} for k in sorted(wk)]

    # 급등 1분 종가 기준 이후
    by_m = {}
    for e in ev:
        by_m.setdefault(e['market'], []).append(e)
    after = {30: [], 240: []}
    for m, es in by_m.items():
        b = P.load(m)
        for e in es:
            i = e['i']
            for k in after:
                if i + k < P.N:
                    after[k].append((b['c'][i + k] / b['c'][i] - 1) * 100)
    x['after_from_spike_close'] = {k: {'above_share': float(np.mean(np.array(v) > 0) * 100), 'n': len(v),
                                       'p25': float(np.percentile(v, 25)), 'median': float(np.median(v)),
                                       'p75': float(np.percentile(v, 75))} for k, v in after.items()}

    # 신호 겹침 적중률 (5분이 아니라 15분 간격 표본 다시 만듦)
    rows = []
    for m in by_m.keys() | set():
        pass
    markets = sorted(p.stem for p in P.DATA.glob('*.npz') if p.stem not in P.PEGGED)
    ev_idx = {}
    for e in ev:
        ev_idx.setdefault(e['market'], []).append(e['i'])
    combos = {'1시간 +3% 이상': 0, '1시간 +3% 이상 · 직전 1시간 거래 빈 분 30% 미만': 0,
              '1시간 +3% 이상 · 15분 거래대금 평소 5배 이상': 0, '셋 다': 0}
    hits = {k: 0 for k in combos}
    base_n = base_hit = 0
    for m in markets:
        b = P.load(m)
        if b is None:
            continue
        nxt = np.zeros(P.N + 61, dtype=bool)
        for k in ev_idx.get(m, []):
            nxt[max(0, k - 60):k + 1] = True
        for j in range(max(P.DAY, b['first'] + P.DAY), P.N - 60, 15):
            f = P.pre_features(b, j)
            if not f:
                continue
            base_n += 1
            base_hit += nxt[j]
            up = f['r60'] >= 3
            busy = f['idle60'] < 30
            loud = f['vr15'] >= 5
            for k, cond in (('1시간 +3% 이상', up), ('1시간 +3% 이상 · 직전 1시간 거래 빈 분 30% 미만', up and busy),
                            ('1시간 +3% 이상 · 15분 거래대금 평소 5배 이상', up and loud), ('셋 다', up and busy and loud)):
                if cond:
                    combos[k] += 1
                    hits[k] += nxt[j]
    x['combo'] = {'base_rate': base_hit / base_n * 100,
                  'rows': [{'signal': k, 'n': combos[k], 'rate': hits[k] / combos[k] * 100 if combos[k] else None} for k in combos]}
    (OUT / 'probe18_extras.json').write_text(json.dumps(x, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(x, ensure_ascii=False, indent=1)[:4000])


if __name__ == '__main__':
    main()
