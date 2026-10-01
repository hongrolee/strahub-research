"""probe18 보고서: outputs/probe18_spike_traits.json + probe18_extras.json → outputs/spike_traits_report.html"""
import json
from datetime import datetime, timedelta
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'
R = json.loads((OUT / 'probe18_spike_traits.json').read_text(encoding='utf-8'))
X = json.loads((OUT / 'probe18_extras.json').read_text(encoding='utf-8'))
M = R['meta']


def f1(v):
    return f'{v:,.1f}'


def f2(v):
    return f'{v:,.2f}'


# ── 그림: 세로 막대 ──
def vbars(labels, values, *, title_x, title_y, unit='', ref=None, ref_label='', highlight=None, w=720, h=260, fmt=f1, every=1):
    pad_l, pad_r, pad_t, pad_b = 52, 16, 18, 52
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b
    top = max(values + ([ref] if ref else [])) * 1.15 or 1
    step = nice_step(top)
    top = step * (int(top / step) + 1)
    n = len(values)
    bw = iw / n
    g = [f'<svg viewBox="0 0 {w} {h}" role="img" class="chart" aria-label="{escape(title_y)}">']
    for k in range(int(top / step) + 1):
        y = pad_t + ih - ih * (k * step) / top
        g.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                 f'<text x="{pad_l - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">{fmt(k * step).rstrip("0").rstrip(".") if "." in fmt(k * step) else fmt(k * step)}</text>')
    for i, (lab, v) in enumerate(zip(labels, values)):
        bh = ih * v / top
        x = pad_l + i * bw + bw * 0.15
        cls = 'bar hot' if highlight and highlight(i, lab, v) else 'bar'
        g.append(f'<rect x="{x:.1f}" y="{pad_t + ih - bh:.1f}" width="{bw * 0.7:.1f}" height="{bh:.1f}" class="{cls}"><title>{escape(str(lab))}: {fmt(v)}{unit}</title></rect>')
        if i % every == 0:
            g.append(f'<text x="{x + bw * 0.35:.1f}" y="{pad_t + ih + 16}" class="tick" text-anchor="middle">{escape(str(lab))}</text>')
    if ref:
        y = pad_t + ih - ih * ref / top
        g.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{y:.1f}" y2="{y:.1f}" class="refline"/>'
                 f'<text x="{w - pad_r}" y="{y - 5:.1f}" class="reflabel" text-anchor="end">{escape(ref_label)}</text>')
    g.append(f'<text x="{pad_l + iw / 2}" y="{h - 8}" class="axis" text-anchor="middle">가로축: {escape(title_x)}</text>')
    g.append(f'<text x="14" y="{pad_t + ih / 2}" class="axis" text-anchor="middle" transform="rotate(-90 14 {pad_t + ih / 2})">세로축: {escape(title_y)}</text>')
    g.append('</svg>')
    return ''.join(g)


def nice_step(top):
    for s in (0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100, 200, 500):
        if top / s <= 6:
            return s
    return 1000


# ── 그림: 가로 막대 (그룹 비교) ──
def hbars(rows, *, title_x, unit='', w=720, fmt=f2, ref=None, ref_label=''):
    """rows: [(라벨, 값, 보조 글자)]"""
    lab_w, pad_r, row_h = 150, 90, 30
    h = 30 + row_h * len(rows) + 40
    iw = w - lab_w - pad_r
    top = max([r[1] for r in rows] + ([ref] if ref else [])) * 1.1 or 1
    g = [f'<svg viewBox="0 0 {w} {h}" role="img" class="chart" aria-label="{escape(title_x)}">']
    for i, (lab, v, note) in enumerate(rows):
        y = 20 + i * row_h
        bw = iw * v / top
        g.append(f'<text x="{lab_w - 10}" y="{y + 15}" class="label" text-anchor="end">{escape(lab)}</text>'
                 f'<rect x="{lab_w}" y="{y + 3}" width="{bw:.1f}" height="17" class="bar"/>'
                 f'<text x="{lab_w + bw + 6:.1f}" y="{y + 15}" class="value">{fmt(v)}{unit}</text>'
                 + (f'<text x="{w - 4}" y="{y + 15}" class="tick" text-anchor="end">{escape(note)}</text>' if note else ''))
    if ref:
        x = lab_w + iw * ref / top
        g.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="14" y2="{20 + row_h * len(rows)}" class="refline"/>'
                 f'<text x="{x + 4:.1f}" y="12" class="reflabel">{escape(ref_label)}</text>')
    g.append(f'<text x="{lab_w + iw / 2}" y="{h - 8}" class="axis" text-anchor="middle">가로축: {escape(title_x)} · 세로축: 묶음</text></svg>')
    return ''.join(g)


# ── 숫자 준비 ──
start = datetime.strptime(M['window_kst'][0][:10], '%Y-%m-%d')
end = datetime.strptime(M['window_kst'][1][:10], '%Y-%m-%d')
weekly = []
for w in X['alone_weekly']:
    ws = datetime.strptime(f"{start.year}-{w['week']}", '%Y-%m-%d')
    days = sum(1 for k in range(7) if start <= ws + timedelta(days=k) <= end)
    weekly.append((w['week'].replace('-', '/'), w['events'] / max(days, 1), days))
peak_week = max(weekly, key=lambda x: x[1])
alone = X['alone_events']
hour = X['alone_hour']
wd = X['alone_weekday']
WD = ['월', '화', '수', '목', '금', '토', '일']
pre = R['pre']
med = lambda f, side: pre[f][side]['median']
base_rate = R['hit_base']
r60 = R['hit']['r60']
combo = {row['signal']: row['rate'] for row in X['combo']['rows']}
after_pre = R['after']
after_spk = X['after_from_spike_close']
conc = R['concentration']
grp = lambda key: [(g['group'], g['per30_mean'], f"코인 {g['coins']}개") for g in R[key]]
by_listing = [g for g in R['by_listing'] if g['group'] != '알 수 없음']
lst = {g['group']: g['per30_mean'] for g in by_listing}
prc = {g['group']: g['per30_mean'] for g in R['by_price']}
bnc = {g['group']: g['per30_mean'] for g in R['by_binance']}
val = {g['group']: g['per30_mean'] for g in R['by_value']}
warn = {g['group']: g['per30_mean'] for g in R['by_warning']}
rep = R['repeat']
r60_3 = next(r for r in r60 if r['from'] == 3)
r60_hi = [r for r in r60 if (r['from'] or -99) >= 3]
up3_rate = combo['1시간 +3% 이상']
nine = X['nine_oclock']

hour_chart = vbars([f'{h}' for h in range(24)], hour, title_x='하루 중 시각 (한국 시간, 시)', title_y='급등 비율 (%)', unit='%',
                   ref=100 / 24, ref_label='아무 때나 고르게 났다면 4.2%', highlight=lambda i, l, v: v > 100 / 24 * 2)
week_chart = vbars([w[0] for w in weekly], [w[1] for w in weekly], title_x='주 (월요일 날짜)', title_y='하루 평균 급등 수 (건)',
                   unit='건', highlight=lambda i, l, v: v >= 6)
wd_rate = [v / (sum(wd) / 7) * 100 for v in wd]
wd_chart = vbars(WD, [v for v in wd], title_x='요일', title_y='급등 수 (건, 90일 합계)', unit='건', w=420, h=230,
                 ref=sum(wd) / 7, ref_label='평균', highlight=lambda i, l, v: i >= 5, fmt=lambda v: f'{v:,.0f}')
hit_rows = [(('' if r['from'] is None else f"{r['from']:+g}%") + ' ~ ' + ('' if r['to'] is None else f"{r['to']:+g}%"), r['rate'],
             f"{r['n']:,}번 중") for r in r60]
hit_chart = hbars(hit_rows, title_x='다음 60분 안에 순간 급등이 난 비율 (%)', unit='%', fmt=lambda v: f'{v:.2f}',
                  ref=base_rate, ref_label=f'평소 {base_rate:.2f}%')
list_chart = hbars([(g['group'], g['per30_mean'], f"코인 {g['coins']}개") for g in by_listing], title_x='코인 하나당 30일에 급등 수 (건)', unit='건')
price_chart = hbars(grp('by_price'), title_x='코인 하나당 30일에 급등 수 (건)', unit='건')
value_chart = hbars(grp('by_value'), title_x='코인 하나당 30일에 급등 수 (건)', unit='건')
bin_chart = hbars(grp('by_binance') + grp('by_warning'), title_x='코인 하나당 30일에 급등 수 (건)', unit='건')


def pre_card(label, ev, ct, unit, note):
    return (f'<div class="pre"><div class="pre-label">{label}</div>'
            f'<div class="pre-nums"><span class="pre-ev">{ev}{unit}</span><span class="pre-vs">평소</span><span class="pre-ct">{ct}{unit}</span></div>'
            f'<p>{note}</p></div>')


pre_cards = ''.join([
    pre_card('직전 15분 거래대금 (그 코인 평소의 몇 배)', f"{med('vr15', 'spike'):.1f}", f"{med('vr15', 'control'):.1f}", '배',
             '급등 직전 15분에 이미 거래가 평소보다 크게 늘어 있었습니다.'),
    pre_card('직전 1시간 가격 변화', f"{med('r60', 'spike'):+.1f}", f"{med('r60', 'control'):+.1f}", '%',
             '튀기 전 한 시간 동안 이미 조금씩 오르고 있던 경우가 많았습니다.'),
    pre_card('직전 4시간 가격 폭 (가장 높은 값과 낮은 값의 차이)', f"{med('rng240', 'spike'):.1f}", f"{med('rng240', 'control'):.1f}", '%',
             '조용히 멈춰 있던 코인보다, 이미 크게 출렁이던 코인이 튀었습니다.'),
    pre_card('직전 1시간 중 거래가 한 건도 없던 분', f"{med('idle60', 'spike'):.0f}", f"{med('idle60', 'control'):.0f}", '%',
             '거래가 뜸하던 코인이 갑자기 튀는 경우보다, 거래가 붙어 있던 코인이 튀는 경우가 많았습니다.'),
])

top10 = ''.join(f'<tr><td>{escape(n or m)}</td><td class="num">{c}건</td></tr>' for m, n, c in conc['top10'])

html = f'''<title>순간 급등 코인 분석</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+KR:wght@400;500;700&display=swap">
<style>
/* 레이아웃: 한 줄 읽기 칸(약 70자) + 그림은 칸 너비 가득. 시세 화면처럼 오름=빨강, 비교 기준=파랑 */
:root {{
  --bg: #f4f5f7; --surface: #ffffff; --ink: #18202c; --muted: #5b6474; --line: #dde1e8;
  --spike: #d4473b; --calm: #3b6db3; --soft: #fbeceb;
  --sans: "IBM Plex Sans KR", "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Consolas, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #10141b; --surface: #171d26; --ink: #e6eaf0; --muted: #9aa3b2; --line: #2a3240;
  --spike: #ff6d60; --calm: #74a2e6; --soft: #2a1b1d; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #10141b; --surface: #171d26; --ink: #e6eaf0; --muted: #9aa3b2; --line: #2a3240;
  --spike: #ff6d60; --calm: #74a2e6; --soft: #2a1b1d; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--ink); font-family: var(--sans); font-size: 16px; line-height: 1.75; }}
.wrap {{ max-width: 760px; margin: 0 auto; padding-inline: 18px; padding-block: 40px 72px; display: grid; gap: 30px; }}
header {{ display: grid; gap: 10px; }}
.eyebrow {{ font-family: var(--mono); font-size: 12px; letter-spacing: .08em; color: var(--muted); text-transform: uppercase; }}
h1 {{ font-size: 32px; line-height: 1.3; margin: 0; text-wrap: balance; letter-spacing: -.01em; }}
h2 {{ font-size: 22px; margin: 0; text-wrap: balance; }}
p {{ margin: 0; }}
.meta {{ font-size: 14px; color: var(--muted); }}
.tldr {{ background: var(--surface); border: 1px solid var(--line); border-left: 4px solid var(--spike); border-radius: 6px; padding: 18px 20px; }}
.tldr ol {{ margin: 0; padding-left: 20px; display: grid; gap: 8px; }}
section {{ display: grid; gap: 14px; }}
.oneline {{ font-weight: 700; background: var(--soft); border-radius: 6px; padding: 10px 14px; }}
.chart {{ width: 100%; height: auto; display: block; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; }}
.chart .grid {{ stroke: var(--line); stroke-width: 1; }}
.chart .bar {{ fill: var(--calm); }}
.chart .bar.hot {{ fill: var(--spike); }}
.chart .refline {{ stroke: var(--muted); stroke-dasharray: 4 4; stroke-width: 1.2; }}
.chart text {{ fill: var(--muted); font-family: var(--mono); font-size: 11px; }}
.chart .axis {{ fill: var(--ink); font-family: var(--sans); font-size: 12px; }}
.chart .label {{ fill: var(--ink); font-family: var(--sans); font-size: 13px; }}
.chart .value {{ fill: var(--ink); font-size: 12px; font-weight: 600; }}
.chart .reflabel {{ fill: var(--muted); font-family: var(--sans); font-size: 11px; }}
.pres {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; }}
.pre {{ background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 14px 16px; display: grid; gap: 6px; min-width: 0; }}
.pre-label {{ font-size: 13px; color: var(--muted); }}
.pre-nums {{ display: flex; align-items: baseline; gap: 10px; font-family: var(--mono); font-variant-numeric: tabular-nums; }}
.pre-ev {{ font-size: 26px; font-weight: 600; color: var(--spike); }}
.pre-vs {{ font-size: 12px; color: var(--muted); }}
.pre-ct {{ font-size: 18px; color: var(--calm); }}
.pre p {{ font-size: 14px; }}
.note {{ font-size: 14px; color: var(--muted); }}
details {{ background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 10px 14px; }}
summary {{ cursor: pointer; font-weight: 500; }}
summary:focus-visible {{ outline: 2px solid var(--calm); outline-offset: 2px; }}
.tbl {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; margin-top: 8px; }}
td, th {{ border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left; }}
.num {{ font-family: var(--mono); font-variant-numeric: tabular-nums; text-align: right; }}
.two {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; align-items: start; }}
.two > * {{ min-width: 0; }}
ul.plain {{ margin: 0; padding-left: 20px; display: grid; gap: 6px; }}
footer {{ font-size: 13px; color: var(--muted); border-top: 1px solid var(--line); padding-top: 16px; display: grid; gap: 6px; }}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">StraHub 연구 · probe18 · 내부 보고서</div>
  <h1>순간 급등은 언제, 어떤 코인에서, 무슨 신호 뒤에 났을까</h1>
  <p class="meta">업비트 원화 시장 {M['coins']}개 코인 · {M['window_kst'][0][:10]} ~ {M['window_kst'][1][:10]} (90일) · 1분봉 · 순간 급등 {M['events']}건</p>
</header>

<div class="tldr">
  <ol>
    <li><b>언제</b>: 코인 혼자 튄 급등의 {hour[9]:.0f}%가 <b>아침 9시대</b>에 났고, 그중 {nine['events']}건은 9시 정각이었습니다. 주말(토·일)이 평일보다 많았고, 8월 중순부터 늘어 9월 중순에 가장 많았습니다.</li>
    <li><b>튀기 전</b>: 이미 거래가 평소보다 늘고({med('vr15', 'spike'):.0f}배) 값이 조금씩 오르던(1시간 {med('r60', 'spike'):+.1f}%) 코인이 튀었습니다. 다만 그런 신호가 있어도 실제로 한 시간 안에 튄 경우는 100번 중 1~2번이었습니다.</li>
    <li><b>어떤 코인</b>: 상장 3개월이 안 된 코인, 1원 미만·10원 미만의 싼 코인, 바이낸스에 없는 코인이 다른 코인보다 2~3배 자주 튀었습니다. 하루 거래대금 300억 원이 넘는 큰 코인은 한 번도 없었습니다.</li>
  </ol>
</div>

<section>
  <h2>무엇을 순간 급등으로 셌나</h2>
  <p>StraBot의 "순간 급등" 조건과 같은 계산입니다. 방금 끝난 1분 동안 거래대금이 1천만 원 이상이면서 그 코인의 직전 60분 평균보다 5배 넘게 몰리고, 그 1분 사이에 값이 5% 넘게 오른 때를 셌습니다. 봇은 여기에 "직전 1시간 +10%"를 더 보는데, 이번에는 급등 자체를 넓게 보려고 그 조건은 뺐습니다. 같은 코인에서 30분 안에 이어진 급등은 한 건으로 셌습니다.</p>
  <p>이렇게 센 급등은 {M['events']}건이고, 그중 봇 조건까지 맞은 것은 {sum(e['bot'] for e in R['events'])}건입니다. 예전에 봇 검증에 쓴 급등 52건(같은 기간)은 이번 계산에서도 모두 잡혔습니다. {M['coins']}개 코인 가운데 {conc['coins_with_event']}개({conc['coins_with_event'] / M['coins'] * 100:.0f}%)가 90일 동안 한 번 이상 튀었습니다.</p>
</section>

<section>
  <h2>최근 정말 늘었나</h2>
  <p class="oneline">한 줄로 말하면, 늘었습니다. 7월에는 하루 1~4건이었는데 8월 중순부터 늘어 {peak_week[0]} 주에는 하루 {peak_week[1]:.1f}건까지 났고, 9월 말에는 조금 줄었습니다.</p>
  {week_chart}
  <p class="note">8월 22일 오후 2시 12분에 시장 전체가 한꺼번에 뛰어 42개 코인이 같은 1분에 튄 일이 있었습니다. 이런 날 하루가 숫자를 부풀리지 않도록, 이 그림과 아래 시간대·요일 그림은 같은 10분 안에 다섯 개 넘게 함께 튄 급등({X['wide_events']}건)을 빼고 그렸습니다. 첫 주와 마지막 주는 며칠만 들어 있어서 하루 평균으로 맞췄습니다.</p>
</section>

<section>
  <h2>언제 잘 났나</h2>
  <p class="oneline">한 줄로 말하면, 아침 9시가 압도적이었고 주말에 더 많았습니다. 비트코인이 움직인 뒤에 따라 튄 경우는 드물었습니다.</p>
  {hour_chart}
  <p>코인 혼자 튄 급등의 {nine['hour9_share']:.0f}%가 9시대에 났습니다. 그중 {nine['events']}건은 9시 0분 정각이었고, 이 정각 급등은 특정한 며칠이 아니라 {nine['days']}일에 걸쳐 고르게 나왔습니다. 업비트는 매일 아침 9시에 "전일 대비" 등락을 새로 세기 시작하는데, 그 시각에 맞춰 주문이 몰리는 것으로 보입니다. 다만 이번 자료만으로 이유를 확인할 수는 없습니다. 90일을 앞뒤 절반으로 나눠 봐도 9시대 비율은 {R['hour_first_half'][9]:.0f}%와 {R['hour_second_half'][9]:.0f}%로 같은 모습이었습니다.</p>
  <div class="two">
    {wd_chart}
    <div style="display:grid; gap:10px;">
      <p>토요일 {wd[5]}건, 일요일 {wd[6]}건으로 평일(하루 {min(wd[:5])}~{max(wd[:5])}건)보다 많았습니다.</p>
      <p>급등 직전 1시간 동안 비트코인은 보통 거의 움직이지 않았습니다(중앙값 {R['btc_1h']['median']:+.2f}%). 즉 대부분의 순간 급등은 시장 흐름과 상관없이 그 코인 혼자 일어났습니다. 다만 8월 22일처럼 시장 전체가 한꺼번에 뛰는 날에는 수십 개가 동시에 튀었습니다.</p>
    </div>
  </div>
</section>

<section>
  <h2>튀기 직전에는 무슨 일이 있었나</h2>
  <p class="oneline">한 줄로 말하면, 조용하던 코인이 갑자기 튄 게 아니라 이미 거래가 붙고 조금씩 오르던 코인이 튀었습니다. 하지만 그 신호만으로 맞히기는 어렵습니다.</p>
  <p>같은 코인, 같은 시간대에서 급등이 없던 시각({M['controls']:,}곳)을 골라 급등 직전과 비교했습니다. 빨간 숫자가 급등 직전, 파란 숫자가 평소의 중앙값입니다.</p>
  <div class="pres">{pre_cards}</div>
  <p>그렇다면 이런 신호가 보일 때 실제로 곧 튀었을까요? 모든 코인의 90일을 15분마다 살펴 "직전 1시간 가격 변화"별로 다음 60분 안에 급등이 난 비율을 셌습니다.</p>
  {hit_chart}
  <p>직전 1시간에 3% 넘게 오른 때는 평소({base_rate:.2f}%)보다 훨씬 자주 튀었습니다(+3~+5%일 때 {r60_3['rate']:.1f}%, +10% 넘게 오른 때 {r60[-1]['rate']:.1f}%). 그래도 비율로 보면 여전히 작습니다. 1시간 +3% 이상인 때 전체로 보면 {up3_rate:.1f}%, 거래대금이 평소의 5배 넘게 몰리고 거래가 빈 분도 적은 조건까지 겹쳐도 {combo['셋 다']:.1f}%였습니다. <b>신호가 보인 100번 중 98번은 한 시간 안에 아무 일도 없었다는 뜻</b>입니다.</p>
</section>

<section>
  <h2>어떤 코인이 잘 튀나</h2>
  <p class="oneline">한 줄로 말하면, 새로 상장한 코인, 값이 싼 코인, 바이낸스에 없는 코인, 거래대금이 중간보다 작은 코인이 자주 튀었습니다.</p>
  <p>코인마다 데이터가 있는 기간이 달라서(기간 중 상장한 코인 등) "코인 하나당 30일에 몇 번 튀었나"로 맞춰 비교했습니다.</p>
  <h3 style="margin:0; font-size:17px;">상장한 지 얼마나 됐나</h3>
  {list_chart}
  <p>상장 3개월이 안 된 코인은 30일에 {lst['3개월 미만']:.1f}번 튀어, 오래된 코인({min(v for k, v in lst.items() if k != '3개월 미만'):.1f}~{max(v for k, v in lst.items() if k != '3개월 미만'):.1f}번)의 약 {lst['3개월 미만'] / (sum(v for k, v in lst.items() if k != '3개월 미만') / (len(lst) - 1)):.0f}배였습니다.</p>
  <h3 style="margin:0; font-size:17px;">가격대</h3>
  {price_chart}
  <p>1원 미만 코인은 30일에 {prc['1원 미만']:.1f}번, 1만 원 이상 코인은 {prc['1만원 이상']:.2f}번이었습니다. 값이 쌀수록 자주 튀는 모습이 뚜렷했습니다.</p>
  <h3 style="margin:0; font-size:17px;">하루 거래대금 규모</h3>
  {value_chart}
  <p>하루 거래대금 10억~30억 원인 코인이 가장 자주 튀었고({val['10억~30억']:.2f}번), 100억 원이 넘는 코인은 드물었으며 300억 원 넘는 코인은 한 번도 없었습니다. 10억 원 미만 코인이 조금 적은 것은 "1분 거래대금 1천만 원 이상"이라는 조건을 넘기기 어렵기 때문으로 보입니다.</p>
  <h3 style="margin:0; font-size:17px;">바이낸스 상장 여부 · 유의 표시</h3>
  {bin_chart}
  <p>바이낸스에 없는 코인(업비트 중심 코인)은 30일에 {bnc['바이낸스에 없음']:.2f}번으로, 바이낸스에도 있는 코인({bnc['바이낸스에도 있음']:.2f}번)의 약 {bnc['바이낸스에 없음'] / bnc['바이낸스에도 있음']:.1f}배였습니다. 유의·주의 표시가 있는 코인도 {warn['유의·주의 표시 있음']:.1f}번으로 많았는데, 이 표시는 <b>자료를 받은 날(10월 2일) 기준</b>이라 급등 때문에 나중에 붙은 표시일 수도 있습니다. 원인으로 읽으면 안 됩니다.</p>
  <h3 style="margin:0; font-size:17px;">한 번 튄 코인은 또 튀나</h3>
  <p>앞 45일에 튄 코인 {rep['first_half_coins']}개 중 {rep['again_share']:.0f}%가 뒤 45일에도 튀었고, 앞에서 안 튄 코인은 {rep['others_share']:.0f}%였습니다. 조금 더 잘 튀지만 큰 차이는 아니었습니다. 가장 많이 튄 10개 코인이 전체 급등의 {conc['top10_share']:.0f}%를 차지했습니다.</p>
  <details><summary>가장 많이 튄 코인 10개 보기</summary><div class="tbl"><table><thead><tr><th>코인</th><th class="num">90일 급등</th></tr></thead><tbody>{top10}</tbody></table></div></details>
</section>

<section>
  <h2>참고: 튄 뒤에는</h2>
  <p class="oneline">한 줄로 말하면, 급등 전에 들고 있던 사람에게는 대체로 좋았고, 급등을 보고 따라 산 사람에게는 대체로 나빴습니다.</p>
  <div class="tbl"><table>
    <thead><tr><th>비교 기준 가격</th><th class="num">30분 뒤 더 높았던 비율</th><th class="num">4시간 뒤 더 높았던 비율</th><th class="num">4시간 뒤 중앙값</th></tr></thead>
    <tbody>
      <tr><td>급등 직전 값 (미리 들고 있던 경우)</td><td class="num">{after_pre['a30']['above_share']:.0f}%</td><td class="num">{after_pre['a240']['above_share']:.0f}%</td><td class="num">{after_pre['a240']['median']:+.1f}%</td></tr>
      <tr><td>급등 1분이 끝난 값 (보고 따라 산 경우)</td><td class="num">{after_spk['30']['above_share']:.0f}%</td><td class="num">{after_spk['240']['above_share']:.0f}%</td><td class="num">{after_spk['240']['median']:+.1f}%</td></tr>
    </tbody></table></div>
  <p>급등을 확인한 뒤 산 가격 기준으로는 30분 뒤 {after_spk['30']['above_share']:.0f}%, 4시간 뒤 {after_spk['240']['above_share']:.0f}%만 그보다 높았습니다. 수수료와 불리한 체결은 넣지 않은 숫자라 실제로는 더 나쁩니다. 급등 뒤 손익은 블로그 5편과 9편에서 따로 다뤘습니다.</p>
</section>

<section>
  <h2>이 보고서를 읽을 때</h2>
  <ul class="plain">
    <li>90일, 업비트 원화 시장 한 곳의 기록입니다. 다른 기간에도 같으리라는 보장은 없습니다. 기간을 반으로 나눠 봐도 시간대와 직전 신호의 방향은 같았습니다.</li>
    <li>유의·주의 표시와 바이낸스 상장 여부는 자료를 받은 날 기준이고, 상장일은 일봉 기록의 첫날(최대 약 5년 전까지)로 셌습니다.</li>
    <li>호가창(주문 대기 물량)과 뉴스·공지는 보지 못했습니다. 그래서 9시 정각 급등이나 8월 22일 같은 일이 왜 일어났는지는 이 자료로 확인할 수 없습니다.</li>
    <li>있었던 일을 센 통계일 뿐이며, 어떤 코인을 사거나 팔라는 뜻이 아닙니다.</li>
  </ul>
</section>

<footer>
  <div>계산: strahub-research · probe18_fetch.py (1분봉 받기) · probe18_spike_traits.py (급등 찾기·비교) · probe18_extras.py · probe18_report.py</div>
  <div>급등 판정: StraBot bot_core/conditions.py 의 순간 급등과 같은 계산 (직전 1시간 조건만 뺌)</div>
</footer>
</div>
'''
(OUT / 'spike_traits_report.html').write_text(html, encoding='utf-8')
print('written', len(html))
