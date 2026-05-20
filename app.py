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

h1 { color: #5a4fcf !important; font-weight: 700 !important; text-align: center; }
h2, h3 { color: #5a4fcf !important; }

[data-testid="stInfo"] { background: #f0eeff !important; border-left: 4px solid #7c6fe0 !important; border-radius: 8px; }
[data-testid="stSuccess"] { background: #eafaf1 !important; border-left: 4px solid #00a85a !important; border-radius: 8px; }

[data-testid="stMetric"] {
    background: #fff; border: 0.5px solid #e0dff5;
    border-left: 3px solid #7c6fe0; border-radius: 8px; padding: 0.8rem 1rem;
}
[data-testid="stMetricLabel"] { color: #999 !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #5a4fcf !important; font-weight: 600 !important; }

[data-testid="stExpander"] {
    background: #fff !important;
    border: 0.5px solid #7c6fe0 !important;
    border-radius: 10px !important;
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

.val-pos { color: #00a85a; font-weight: 600; }
.val-neg { color: #e03e3e; font-weight: 600; }
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
top_n_picks = st.sidebar.slider("② ピックアップ台数", min_value=1, max_value=15, value=5, step=1)
pattern_strictness = st.sidebar.slider("③ 波形の一致度", min_value=70, max_value=99, value=90, step=1)

# ==========================================
# データ読み込み
# ==========================================
def load_data(sheet_name: str):
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

# 危険台除外
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
recommendations = recommendations[recommendations['7日間合計'] <= target_7day_max].head(top_n_picks)

# ==========================================
# タイトル
# ==========================================
st.markdown(f"# 🎰 {selected_machine_label.replace('🎰 ', '')} 予測AI")
st.markdown(
    f"<p style='text-align:center; color:#aaa; font-size:0.9rem;'>予測基準日：{latest_date.strftime('%Y年%m月%d日')}</p>",
    unsafe_allow_html=True
)

# ==========================================
# AI予測ランキング（expanderをカード風に）
# ==========================================
st.markdown('<div class="section-label">AI予測ランキング</div>', unsafe_allow_html=True)
st.info(f"💡 7日計 {target_7day_max}枚以下の台から勝率上位 {len(recommendations)}台 を表示")

for rank, (_, row) in enumerate(recommendations.iterrows(), 1):
    circle = "🥇" if rank == 1 else str(rank)
    diff_str = f"+{int(row['差枚'])}" if row['差枚'] >= 0 else str(int(row['差枚']))
    sum_str = f"+{int(row['7日間合計'])}" if row['7日間合計'] >= 0 else str(int(row['7日間合計']))
    diff_color = "#00a85a" if row['差枚'] >= 0 else "#e03e3e"
    sum_color = "#00a85a" if row['7日間合計'] >= 0 else "#e03e3e"
    pct = row['明日勝つ確率(%)']
    label = (
        f"{circle}  **{int(row['台番号'])}番台**　"
        f"勝率 {pct:.1f}%　｜　"
        f"7日計 {sum_str}　差枚 {diff_str}"
    )

    with st.expander(label, expanded=(rank == 1)):
        # 勝率
        st.markdown(
            f"<div style='text-align:center; font-size:2.2rem; font-weight:700; color:#00a85a; margin:0.5rem 0;'>{pct:.1f}%</div>"
            f"<div style='text-align:center; font-size:0.82rem; color:#aaa; margin-bottom:1rem;'>明日の勝率予測</div>",
            unsafe_allow_html=True
        )
        # BB/RB/合成確率
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("BB回数", f"{int(row['BB'])}回")
        with c2: st.metric("RB回数", f"{int(row['RB'])}回")
        gos_val = f"1/{row['合成確率']:.0f}" if row['合成確率'] > 0 else "-"
        with c3: st.metric("合成確率", gos_val)

        # 過去7日間の差枚グラフ
        machine_num = row['台番号']
        machine_history = df[df['台番号'] == machine_num].sort_values('日付')
        # 最新日より前の7日分を取得
        past_history = machine_history[machine_history['日付'] < latest_date].tail(7)

        if len(past_history) > 0:
            chart_data = pd.DataFrame({
                '日付': past_history['日付'].dt.strftime('%m/%d').tolist(),
                '差枚': past_history['差枚'].tolist(),
                '色': ['プラス' if v >= 0 else 'マイナス' for v in past_history['差枚'].tolist()]
            })
            bars = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X('日付:O', sort=None, axis=alt.Axis(labelAngle=0, title=None)),
                y=alt.Y('差枚:Q', axis=alt.Axis(title=None)),
                color=alt.Color('色:N', scale=alt.Scale(
                    domain=['プラス', 'マイナス'],
                    range=['#00a85a', '#e03e3e']
                ), legend=None)
            ).properties(height=200, title="過去7日間の差枚推移")
            zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
                color='#aaa', strokeDash=[3, 3]
            ).encode(y='y:Q')
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
    "<p style='color:#aaa; font-size:0.88rem;'>過去の爆発台と波の形・揺れ幅・プラスマイナス域が一致する台を抽出します。</p>",
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
    pm_candidates = safe_latest_df[safe_latest_df['7日間合計'] <= target_7day_max]

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
                # 過去波形データ構築
                p_history = df[df['台番号'] == best_hp['machine']].sort_values('日付').reset_index(drop=True)
                t_idx_list = p_history[p_history['日付'] == best_hp['raw_date']].index
                pw_cum, pw_daily = [], []
                if len(t_idx_list) > 0:
                    t_idx = t_idx_list[0]
                    t_rec = p_history.loc[t_idx]
                    pw_diffs = [t_rec['7日前の差枚'],t_rec['6日前の差枚'],t_rec['5日前の差枚'],
                                t_rec['4日前の差枚'],t_rec['3日前の差枚'],t_rec['2日前の差枚'],t_rec['1日前の差枚']]
                    pw_cum = [0]; pw_daily = [0]; p_sum = 0
                    for v in pw_diffs:
                        p_sum += v; pw_cum.append(p_sum); pw_daily.append(v)
                    p_sum += t_rec['差枚']; pw_cum.append(p_sum); pw_daily.append(t_rec['差枚'])
                    if t_idx + 1 < len(p_history):
                        n_diff = p_history.loc[t_idx + 1, '差枚']
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
            label = f"**{m['台番号']}番台**　類似度 {m['類似度']}%　｜　{m['一致した過去の爆発台']} と一致　｜　7日計 {sum_str}"

            with st.expander(label, expanded=False):
                # 爆発台の詳細
                st.markdown(f"**▼ 一致した過去の爆発台：{m['一致した過去の爆発台']}**")
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("差枚（爆発日）", f"{m['past_diff']:+}枚")
                with c2: st.metric("BB回数", f"{m['past_BB']}回")
                with c3: st.metric("RB回数", f"{m['past_RB']}回")

                st.markdown("---")

                # 波形比較グラフ（棒＋線、軸ラベルなし）
                x_labels = ["起点","6日前","5日前","4日前","3日前","2日前","1日前","現在(前日)","★爆発","🚀翌日"]
                cw_diffs = m['cw_diffs']
                cw_cum = [0]; cw_daily = [0]; c_sum = 0
                for v in cw_diffs:
                    c_sum += v; cw_cum.append(c_sum); cw_daily.append(v)
                cw_cum.extend([None, None]); cw_daily.extend([None, None])

                past_label = f"過去: {int(m['past_machine'])}番台"
                curr_label = f"現在: {m['台番号']}番台"

                plot_data = []
                for i, label_x in enumerate(x_labels):
                    if i < len(m['pw_cum']) and m['pw_cum'][i] is not None:
                        plot_data.append({'期間': label_x, '種別': past_label, '日別差枚': m['pw_daily'][i]})
                    if i < len(cw_cum) and cw_cum[i] is not None:
                        plot_data.append({'期間': label_x, '種別': curr_label, '日別差枚': cw_daily[i]})

                df_plot = pd.DataFrame(plot_data)

                base = alt.Chart(df_plot).encode(
                    x=alt.X('期間:O', sort=x_labels,
                            axis=alt.Axis(labelAngle=-30, title=None)),
                    color=alt.Color('種別:N',
                        legend=alt.Legend(title="", orient="top"),
                        scale=alt.Scale(
                            domain=[past_label, curr_label],
                            range=["#7c6fe0", "#00a85a"]
                        ))
                )
                bars = base.mark_bar(opacity=0.6).encode(
                    xOffset='種別:N',
                    y=alt.Y('日別差枚:Q', axis=alt.Axis(title=None, labels=False, ticks=False))
                )
                zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
                    color='#aaa', strokeDash=[3, 3]
                ).encode(y='y:Q')

                chart = alt.layer(bars, zero_line).properties(height=250)
                st.altair_chart(chart, use_container_width=True)
    else:
        st.info(f"波形一致度 {pattern_strictness}% を超える台はありませんでした。")
else:
    st.info("過去の高設定データが不足しています。")
