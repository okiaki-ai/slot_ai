import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import altair as alt

st.set_page_config(page_title="ジャグラー予測AI", layout="wide", page_icon="🎰")

st.markdown("""
<style>
.stApp { background-color: #f8f8fc; color: #1a1a2e; }

[data-testid="stSidebar"] { background: #f0eeff; border-right: 1px solid #e0dff5; }
[data-testid="stSidebar"] * { color: #1a1a2e !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #5a4fcf !important; }
h2, h3 { color: #5a4fcf !important; }

[data-testid="stInfo"] { background: #f0eeff !important; border-left: 4px solid #7c6fe0 !important; border-radius: 8px; }
[data-testid="stSuccess"] { background: #eafaf1 !important; border-left: 4px solid #00a85a !important; border-radius: 8px; }

[data-testid="stMetric"] {
    background: #fff; border: 0.5px solid #e0dff5;
    border-left: 3px solid #7c6fe0; border-radius: 8px; padding: 0.8rem 1rem;
}
[data-testid="stMetricLabel"] { color: #999 !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #5a4fcf !important; font-weight: 600 !important; }

/* expander全体 */
[data-testid="stExpander"] {
    background: #fff !important;
    border: 0.5px solid #e0dff5 !important;
    border-radius: 12px !important;
    margin-bottom: 8px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
    overflow: hidden !important;
}

/* expanderのヘッダー行 */
[data-testid="stExpander"] summary {
    background: #fff !important;
    padding: 12px 16px !important;
    border-radius: 12px !important;
    list-style: none !important;
}
[data-testid="stExpander"] summary::-webkit-details-marker { display: none; }

/* expanderの中身 */
[data-testid="stExpander"] > div[data-testid="stExpanderDetails"] {
    background: #faf9ff !important;
    border-top: 0.5px solid #e0dff5 !important;
    padding: 12px 16px !important;
}

hr { border-color: #e0dff5 !important; }
[data-testid="stSelectbox"] label { color: #5a4fcf !important; font-weight: 600; }

.section-label {
    font-size: 11px; font-weight: 600; color: #aaa;
    text-transform: uppercase; letter-spacing: 0.8px;
    margin: 14px 0 8px;
    display: flex; align-items: center; gap: 8px;
}
.section-label::after { content: ''; flex: 1; border-top: 0.5px solid #e0dff5; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# サイドバー
# ==========================================
st.sidebar.markdown("## ⚙️ 絞り込みフィルター")
st.sidebar.write("数値を動かすと結果がリアルタイムで絞り込まれます。")
st.sidebar.markdown("---")

MACHINE_SHEET_MAP = {
    "🎰 マイジャグラー5": "maijag5",
    "🎰 ゴーゴージャグラー": "gojag",
    "🎰 ハッピージャグラー": "happy",
}
selected_machine_label = st.sidebar.selectbox("🕹️ 機種を選択", list(MACHINE_SHEET_MAP.keys()))
selected_sheet = MACHINE_SHEET_MAP[selected_machine_label]
st.sidebar.markdown("---")
target_7day_max = st.sidebar.slider("① 7日計の上限", min_value=-5000, max_value=2000, value=0, step=100)
target_g_max = st.sidebar.slider("② 基準日の回転数（上限）", min_value=1000, max_value=15000, value=15000, step=100)
target_diff_min = st.sidebar.slider("③ 基準日の差枚（下限）", min_value=-5000, max_value=2000, value=-5000, step=100)
top_n_picks = st.sidebar.slider("④ ピックアップ台数", min_value=1, max_value=15, value=5, step=1)
pattern_strictness = st.sidebar.slider("⑤ 波形の一致度", min_value=70, max_value=99, value=90, step=1)

# ==========================================
# データ読み込み
# ==========================================
def load_data(sheet_name):
    try:
        base_url = st.secrets[f"spreadsheet_url_{sheet_name}"]
    except KeyError:
        st.error(f"❌ secrets に 'spreadsheet_url_{sheet_name}' が見つかりません。")
        st.stop()
    df = pd.read_csv(base_url)
    df = df[df['台番号'].astype(str) != '平均']
    df['G数'] = pd.to_numeric(df['G数'].astype(str).str.replace(",", ""), errors='coerce').fillna(0).astype(int)
    df['差枚'] = pd.to_numeric(df['差枚'].astype(str).str.replace("+", "").str.replace(",", ""), errors='coerce').fillna(0).astype(int)
    df['BB'] = pd.to_numeric(df['BB'], errors='coerce').fillna(0).astype(int)
    df['RB'] = pd.to_numeric(df['RB'], errors='coerce').fillna(0).astype(int)
    df['合成確率_表示用'] = df['合成確率'].astype(str)
    df['BB確率'] = pd.to_numeric(df['BB確率'].astype(str).str.replace("1/", ""), errors='coerce').fillna(0.0)
    df['RB確率'] = pd.to_numeric(df['RB確率'].astype(str).str.replace("1/", ""), errors='coerce').fillna(0.0)
    df['合成確率'] = pd.to_numeric(df['合成確率'].astype(str).str.replace("1/", ""), errors='coerce').fillna(0.0)
    df['日付'] = pd.to_datetime(df['日付'])
    return df.sort_values(['台番号', '日付'])

with st.spinner('データを読み込み中...'):
    df = load_data(selected_sheet)
    latest_date = df['日付'].max()

# ==========================================
# AI学習
# ==========================================
df['日'] = df['日付'].dt.day
df['還元日'] = df['日'].apply(lambda x: 1 if x == 3 or (1 <= x <= 5) or (27 <= x <= 31) else 0)
df['警戒日'] = df['日'].apply(lambda x: 1 if 10 <= x <= 20 else 0)
corner_list = [521, 540, 541, 560]
df['角台'] = df['台番号'].isin(corner_list).astype(int)
for i in range(1, 8):
    df[f'{i}日前の差枚'] = df.groupby('台番号')['差枚'].shift(i).fillna(0)
df['1日前のRB確率'] = df.groupby('台番号')['RB確率'].shift(1).fillna(0.0)
# 7日間合計：最新日を含む直近7日分の差枚合計（表の合計値と一致）
df['7日間合計'] = (df['差枚'] + df['1日前の差枚'] + df['2日前の差枚'] + df['3日前の差枚'] +
                   df['4日前の差枚'] + df['5日前の差枚'] + df['6日前の差枚'])
df['V字回復候補'] = df['1日前の差枚'].apply(lambda x: 1 if -4000 <= x <= -2500 else 0)
df['回収トラップ'] = df['1日前の差枚'].apply(lambda x: 1 if x > 3000 else 0)
df['翌日の差枚'] = df.groupby('台番号')['差枚'].shift(-1)
df['翌日勝つか'] = (df['翌日の差枚'] > 0).astype(int)

train_df = df.dropna(subset=['翌日の差枚'])
features = ['G数','差枚','BB','RB','合成確率','還元日','警戒日','角台','V字回復候補','回収トラップ',
            '1日前の差枚','2日前の差枚','3日前の差枚','4日前の差枚','5日前の差枚','6日前の差枚','7日前の差枚']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_df[features], train_df['翌日勝つか'])

latest_df = df[df['日付'] == latest_date].copy()
exclude_condition = (
    (latest_df['差枚'] >= 1000) |
    (latest_df['1日前の差枚'] >= 1000) |
    (latest_df['2日前の差枚'] >= 1000) |
    ((latest_df['RB確率'] > 0) & (latest_df['RB確率'] <= 310.0)) |
    ((latest_df['1日前のRB確率'] > 0) & (latest_df['1日前のRB確率'] <= 310.0))
)
safe_latest_df = latest_df[~exclude_condition].copy()
safe_latest_df['明日勝つ確率(%)'] = model.predict_proba(safe_latest_df[features])[:, 1] * 100
recommendations = safe_latest_df.sort_values('明日勝つ確率(%)', ascending=False)
recommendations = recommendations[
    (recommendations['7日間合計'] <= target_7day_max) &
    (recommendations['G数'] <= target_g_max) &
    (recommendations['差枚'] >= target_diff_min)
].head(top_n_picks)

# ==========================================
# タイトル
# ==========================================
machine_name = selected_machine_label.replace('🎰 ', '')
st.markdown(
    f"<h1 style='text-align:center; color:#5a4fcf; font-weight:700; font-size:clamp(1.1rem,4vw,1.8rem);'>"
    f"🎰 {machine_name} 予測AI</h1>",
    unsafe_allow_html=True
)
st.markdown(
    f"<p style='text-align:center; color:#aaa; font-size:0.85rem; margin-top:-0.3rem;'>"
    f"予測基準日：{latest_date.strftime('%Y年%m月%d日')}</p>",
    unsafe_allow_html=True
)

# ==========================================
# AI予測ランキング
# ==========================================
st.markdown('<div class="section-label">AI予測ランキング</div>', unsafe_allow_html=True)
st.info(f"💡 7日計 {target_7day_max}枚以下の台から勝率上位 {len(recommendations)}台 を表示")

for rank, (_, row) in enumerate(recommendations.iterrows(), 1):
    machine_id = int(row['台番号'])
    diff_str  = f"+{int(row['差枚'])}"    if row['差枚']      >= 0 else str(int(row['差枚']))
    sum_str   = f"+{int(row['7日間合計'])}" if row['7日間合計'] >= 0 else str(int(row['7日間合計']))
    diff_color = "#00a85a" if row['差枚']      >= 0 else "#e03e3e"
    sum_color  = "#00a85a" if row['7日間合計'] >= 0 else "#e03e3e"
    pct = row['明日勝つ確率(%)']
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    medal = medals.get(rank, f"[{rank}]")

    # expanderのラベルをHTMLで組み立て
    label = (
        f"{medal} {machine_id}番台　"
        f"勝率 {pct:.1f}%　｜　"
        f"7日計 {sum_str}　差枚 {diff_str}"
    )

    with st.expander(label, expanded=False):
        # 勝率
        st.markdown(
            f"<div style='text-align:center; font-size:1.8rem; font-weight:700; color:#00a85a; margin:0.2rem 0;'>{pct:.1f}%</div>"
            f"<div style='text-align:center; font-size:0.78rem; color:#aaa; margin-bottom:0.8rem;'>明日の勝率予測</div>",
            unsafe_allow_html=True
        )
        # 過去7日間テーブル＋グラフ
        machine_history = df[df['台番号'] == row['台番号']].sort_values('日付')
        past_history = machine_history[machine_history['日付'] <= latest_date].tail(7)

        if len(past_history) > 0:
            table_rows = ""
            for _, hr in past_history.iterrows():
                d_color = "#00a85a" if hr['差枚'] >= 0 else "#e03e3e"
                d_sign  = "+" if hr['差枚'] >= 0 else ""
                gos = f"1/{hr['合成確率']:.0f}" if hr['合成確率'] > 0 else "-"
                table_rows += (
                    f"<tr>"
                    f"<td style='text-align:left; color:#888; font-size:11px; padding:6px 8px; border-bottom:0.5px solid #f0eeff;'>{hr['日付'].strftime('%m/%d')}</td>"
                    f"<td style='text-align:right; padding:6px 8px; border-bottom:0.5px solid #f0eeff;'>{int(hr['G数']):,}</td>"
                    f"<td style='text-align:right; padding:6px 8px; border-bottom:0.5px solid #f0eeff;'>{int(hr['BB'])}</td>"
                    f"<td style='text-align:right; padding:6px 8px; border-bottom:0.5px solid #f0eeff;'>{int(hr['RB'])}</td>"
                    f"<td style='text-align:right; padding:6px 8px; border-bottom:0.5px solid #f0eeff;'>{gos}</td>"
                    f"<td style='text-align:right; padding:6px 8px; border-bottom:0.5px solid #f0eeff; color:{d_color}; font-weight:600;'>{d_sign}{int(hr['差枚'])}</td>"
                    f"</tr>"
                )

            st.markdown(
                f"<div style='overflow-x:auto; margin-bottom:12px;'>"
                f"<table style='width:100%; border-collapse:collapse; font-size:12px;'>"
                f"<thead><tr style='background:#f8f8fc;'>"
                f"<th style='text-align:left; padding:6px 8px; color:#aaa; font-weight:500; font-size:10px; border-bottom:0.5px solid #e0dff5;'>日付</th>"
                f"<th style='text-align:right; padding:6px 8px; color:#aaa; font-weight:500; font-size:10px; border-bottom:0.5px solid #e0dff5;'>回転数</th>"
                f"<th style='text-align:right; padding:6px 8px; color:#aaa; font-weight:500; font-size:10px; border-bottom:0.5px solid #e0dff5;'>BB</th>"
                f"<th style='text-align:right; padding:6px 8px; color:#aaa; font-weight:500; font-size:10px; border-bottom:0.5px solid #e0dff5;'>RB</th>"
                f"<th style='text-align:right; padding:6px 8px; color:#aaa; font-weight:500; font-size:10px; border-bottom:0.5px solid #e0dff5;'>合成確率</th>"
                f"<th style='text-align:right; padding:6px 8px; color:#aaa; font-weight:500; font-size:10px; border-bottom:0.5px solid #e0dff5;'>差枚</th>"
                f"</tr></thead>"
                f"<tbody>{table_rows}</tbody>"
                f"</table></div>",
                unsafe_allow_html=True
            )

            chart_data = pd.DataFrame({
                '日付': past_history['日付'].dt.strftime('%m/%d').tolist(),
                '差枚': past_history['差枚'].tolist(),
                '色': ['プラス' if v >= 0 else 'マイナス' for v in past_history['差枚'].tolist()]
            })
            bars = alt.Chart(chart_data).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
                x=alt.X('日付:O', sort=None, axis=alt.Axis(labelAngle=0, title=None)),
                y=alt.Y('差枚:Q', axis=alt.Axis(title=None, labels=False, ticks=False)),
                color=alt.Color('色:N',
                    scale=alt.Scale(domain=['プラス','マイナス'], range=['#00a85a','#e03e3e']),
                    legend=None)
            ).properties(height=160, title="差枚推移")
            zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='#bbb', strokeDash=[3,3]).encode(y='y:Q')
            st.altair_chart(alt.layer(bars, zero_line), use_container_width=True)
        else:
            st.caption("グラフ表示に必要なデータが不足しています。")

# ==========================================
# パターンマッチング予測
# ==========================================
high_setting_days = df[(df['RB確率'] > 0) & (df['RB確率'] <= 290) & (df['差枚'] >= 1000)].copy()

st.markdown("---")
st.markdown('<div class="section-label">パターンマッチング予測</div>', unsafe_allow_html=True)
st.markdown(
    "<p style='color:#aaa; font-size:0.85rem; margin-bottom:0.5rem;'>"
    "過去の爆発台と波の形・揺れ幅・プラスマイナス域が一致する台を抽出します。</p>",
    unsafe_allow_html=True
)

if not high_setting_days.empty:
    historical_patterns = []
    for _, row in high_setting_days.iterrows():
        hw = row[['7日前の差枚','6日前の差枚','5日前の差枚','4日前の差枚','3日前の差枚','2日前の差枚','1日前の差枚']].values.astype(float)
        if np.std(hw) > 0:
            historical_patterns.append({
                'date': row['日付'].strftime('%m/%d'),
                'raw_date': row['日付'],
                'machine': row['台番号'],
                'wave': hw,
                'depth': np.max(hw) - np.min(hw),
                'bottom': np.min(hw),
                'diff': int(row['差枚']),
                'BB': int(row['BB']),
                'RB': int(row['RB']),
            })

    match_results = []
    pm_candidates = safe_latest_df[
        (safe_latest_df['7日間合計'] <= target_7day_max) &
        (safe_latest_df['G数'] <= target_g_max) &
        (safe_latest_df['差枚'] >= target_diff_min)
    ]

    for _, row in pm_candidates.iterrows():
        cw_diffs = [row['6日前の差枚'],row['5日前の差枚'],row['4日前の差枚'],
                    row['3日前の差枚'],row['2日前の差枚'],row['1日前の差枚'],row['差枚']]
        cw = np.array(cw_diffs, dtype=float)
        if np.std(cw) > 0:
            cw_depth = np.max(cw) - np.min(cw)
            cw_bottom = np.min(cw)
            best_score = -1
            best_hp = None
            for hp in historical_patterns:
                score = np.corrcoef(cw, hp['wave'])[0, 1]
                if (not np.isnan(score) and score > best_score and
                    abs(cw_depth - hp['depth']) <= 1000 and abs(cw_bottom - hp['bottom']) <= 1000):
                    best_score = score
                    best_hp = hp
            if best_score >= (pattern_strictness / 100.0) and best_hp is not None:
                p_history = df[df['台番号'] == best_hp['machine']].sort_values('日付').reset_index(drop=True)
                t_idx_list = p_history[p_history['日付'] == best_hp['raw_date']].index
                pw_cum, pw_daily = [], []
                if len(t_idx_list) > 0:
                    t_idx = t_idx_list[0]
                    t_rec = p_history.loc[t_idx]
                    pw_diffs_vals = [t_rec['7日前の差枚'],t_rec['6日前の差枚'],t_rec['5日前の差枚'],
                                     t_rec['4日前の差枚'],t_rec['3日前の差枚'],t_rec['2日前の差枚'],t_rec['1日前の差枚']]
                    pw_cum = [0]; pw_daily = [0]; p_sum = 0
                    for v in pw_diffs_vals:
                        p_sum += float(v); pw_cum.append(p_sum); pw_daily.append(float(v))
                    p_sum += float(t_rec['差枚'])
                    pw_cum.append(p_sum); pw_daily.append(float(t_rec['差枚']))
                    if t_idx + 1 < len(p_history):
                        n_diff = float(p_history.loc[t_idx + 1, '差枚'])
                        p_sum += n_diff; pw_cum.append(p_sum); pw_daily.append(n_diff)
                    else:
                        pw_cum.append(None); pw_daily.append(None)

                match_results.append({
                    '台番号': int(row['台番号']),
                    '類似度': round(best_score * 100, 1),
                    '一致した過去の爆発台': f"{best_hp['date']}の{int(best_hp['machine'])}番台",
                    '現在の7日計': int(row['7日間合計']),
                    'past_machine': best_hp['machine'],
                    'past_diff': best_hp['diff'],
                    'past_BB': best_hp['BB'],
                    'past_RB': best_hp['RB'],
                    'cw_diffs': cw_diffs,
                    'pw_cum': pw_cum,
                    'pw_daily': pw_daily,
                })

    if match_results:
        match_results_sorted = sorted(match_results, key=lambda x: x['類似度'], reverse=True)
        st.success(f"🔥 波形が一致する台を {len(match_results_sorted)}台 発見！")

        for m in match_results_sorted:
            sum_str = f"+{m['現在の7日計']}" if m['現在の7日計'] >= 0 else str(m['現在の7日計'])
            exp_label = f"{m['台番号']}番台　類似度 {m['類似度']}%　｜　{m['一致した過去の爆発台']} と一致　｜　7日計 {sum_str}"

            with st.expander(exp_label, expanded=False):
                st.markdown(f"**一致した過去の爆発台：{m['一致した過去の爆発台']}**")
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("差枚（爆発日）", f"{m['past_diff']:+}枚")
                with c2: st.metric("BB回数", f"{m['past_BB']}回")
                with c3: st.metric("RB回数", f"{m['past_RB']}回")
                st.markdown("---")

                x_labels = ["6日前","5日前","4日前","3日前","2日前","1日前","現在(前日)","★爆発","🚀翌日"]
                cw_diffs = m['cw_diffs']
                cw_cum_vals = []; c_sum = 0
                for v in cw_diffs:
                    c_sum += float(v); cw_cum_vals.append(c_sum)
                cw_cum_vals.extend([None, None])

                pw_cum_trimmed   = m['pw_cum'][1:]  if len(m['pw_cum'])  > 1 else []
                pw_daily_trimmed = m['pw_daily'][1:] if len(m['pw_daily']) > 1 else []

                past_label = f"過去: {int(m['past_machine'])}番台"
                curr_label = f"現在: {m['台番号']}番台"
                bar_data = []; line_data = []

                for i, lx in enumerate(x_labels):
                    if i < len(pw_daily_trimmed) and pw_daily_trimmed[i] is not None:
                        bar_data.append({'期間': lx, '種別': past_label, '日別差枚': float(pw_daily_trimmed[i])})
                    if i < len(cw_diffs):
                        bar_data.append({'期間': lx, '種別': curr_label, '日別差枚': float(cw_diffs[i])})
                    if i < len(pw_cum_trimmed) and pw_cum_trimmed[i] is not None:
                        line_data.append({'期間': lx, '種別': past_label, '累積差枚': float(pw_cum_trimmed[i])})
                    if i < len(cw_cum_vals) and cw_cum_vals[i] is not None:
                        line_data.append({'期間': lx, '種別': curr_label, '累積差枚': float(cw_cum_vals[i])})

                df_bar  = pd.DataFrame(bar_data)
                df_line = pd.DataFrame(line_data)
                color_scale = alt.Scale(domain=[past_label, curr_label], range=["#7c6fe0","#00a85a"])
                x_enc = alt.X('期間:O', sort=x_labels, axis=alt.Axis(labelAngle=-30, title=None))

                no_axis = alt.Axis(title=None, labels=False, ticks=False, domain=False)

                bars_chart = alt.Chart(df_bar).mark_bar(opacity=0.55).encode(
                    x=x_enc, xOffset='種別:N',
                    y=alt.Y('日別差枚:Q', axis=no_axis),
                    color=alt.Color('種別:N', scale=color_scale, legend=None)
                )
                zero_line = alt.Chart(pd.DataFrame({'y':[0]})).mark_rule(color='#bbb', strokeDash=[3,3]).encode(
                    y=alt.Y('y:Q', axis=no_axis)
                )
                lines_chart = alt.Chart(df_line).mark_line(size=2.5).encode(
                    x=x_enc,
                    y=alt.Y('累積差枚:Q', axis=no_axis),
                    color=alt.Color('種別:N', scale=color_scale, legend=alt.Legend(title="", orient="top"))
                )
                points_chart = alt.Chart(df_line).mark_circle(size=55, opacity=1).encode(
                    x=x_enc,
                    y=alt.Y('累積差枚:Q', axis=no_axis),
                    color=alt.Color('種別:N', scale=color_scale, legend=None)
                )
                chart = alt.layer(bars_chart, zero_line, lines_chart, points_chart).resolve_scale(
                    y='independent'
                ).properties(height=260)
                st.altair_chart(chart, use_container_width=True)
    else:
        st.info(f"波形一致度 {pattern_strictness}% を超える台はありませんでした。")
else:
    st.info("過去の高設定データが不足しています。")
