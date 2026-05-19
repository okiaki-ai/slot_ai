import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import altair as alt

# ==========================================
# 画面の基本設定
# ==========================================
st.set_page_config(page_title="ジャグラー予測AI", layout="wide", page_icon="🎰")

# ==========================================
# カスタムCSS（デザイン強化）
# ==========================================
st.markdown("""
<style>
/* 全体背景 */
.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 50%, #16213e 100%);
    color: #e0e0e0;
}

/* サイドバー */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #0d0d1a 100%);
    border-right: 1px solid #f0a500;
}
[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}

/* サイドバーヘッダー */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f0a500 !important;
}

/* メインタイトル */
h1 {
    background: linear-gradient(90deg, #f0a500, #ff6b6b, #f0a500);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem !important;
    font-weight: 900 !important;
    text-align: center;
    padding: 0.5rem 0;
}

/* サブヘッダー */
h2, h3 {
    color: #f0a500 !important;
    border-bottom: 1px solid #f0a50055;
    padding-bottom: 0.3rem;
}

/* infoボックス */
[data-testid="stInfo"] {
    background: #1a2a4a !important;
    border-left: 4px solid #4a9eff !important;
    color: #cce0ff !important;
    border-radius: 8px;
}

/* successボックス */
[data-testid="stSuccess"] {
    background: #1a3a2a !important;
    border-left: 4px solid #00e676 !important;
    color: #b9f6ca !important;
    border-radius: 8px;
}

/* dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #f0a50055;
    border-radius: 10px;
    overflow: hidden;
}

/* メトリクス */
[data-testid="stMetric"] {
    background: #1a1a2e;
    border: 1px solid #f0a50066;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
}
[data-testid="stMetricLabel"] {
    color: #aaaaaa !important;
    font-size: 0.8rem !important;
}
[data-testid="stMetricValue"] {
    color: #f0a500 !important;
    font-size: 1.4rem !important;
    font-weight: bold !important;
}

/* expander */
[data-testid="stExpander"] {
    background: #1a1a2e !important;
    border: 1px solid #f0a50044 !important;
    border-radius: 10px !important;
}

/* ボタン・スライダーのアクセントカラー */
[data-testid="stSlider"] > div > div > div {
    background: #f0a500 !important;
}

/* セパレーター */
hr {
    border-color: #f0a50033 !important;
}

/* selectbox */
[data-testid="stSelectbox"] label {
    color: #f0a500 !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# サイドバー
# ==========================================
st.sidebar.markdown("## ⚙️ 絞り込みフィルター")
st.sidebar.markdown("数値を動かすと結果がリアルタイムで絞り込まれます。")
st.sidebar.markdown("---")

# --- 機種選択 ---
MACHINE_SHEET_MAP = {
    "🎰 マイジャグラー5": "maijag5",
    "🎰 ゴーゴージャグラー": "gojag",
    "🎰 ハッピージャグラー": "happy",
}
selected_machine_label = st.sidebar.selectbox(
    "🕹️ 機種を選択",
    list(MACHINE_SHEET_MAP.keys())
)
selected_sheet = MACHINE_SHEET_MAP[selected_machine_label]

st.sidebar.markdown("---")
target_7day_max = st.sidebar.slider("① 7日計の上限 (凹み台狙い)", min_value=-5000, max_value=2000, value=0, step=100)
top_n_picks = st.sidebar.slider("② ピックアップ台数", min_value=1, max_value=15, value=5, step=1)
pattern_strictness = st.sidebar.slider("③ 波形の一致度 (高いほど厳密)", min_value=70, max_value=99, value=90, step=1)

# ==========================================
# メインタイトル
# ==========================================
st.markdown(f"# 🎰 {selected_machine_label.replace('🎰 ', '')} 狙い目予測AI")
st.markdown(
    "<p style='text-align:center; color:#aaaaaa; font-size:0.95rem;'>"
    "過去データに基づき、明日の勝率予測と高設定投入パターンの分析を行います。"
    "</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# ==========================================
# 1. データの読み込み
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

with st.spinner(f'【{selected_machine_label}】のデータを読み込み中...'):
    df = load_data(selected_sheet)
    latest_date = df['日付'].max()

# ==========================================
# 2. AI学習
# ==========================================
df['日'] = df['日付'].dt.day
df['還元日'] = df['日'].apply(lambda x: 1 if x == 3 or (1 <= x <= 5) or (27 <= x <= 31) else 0)
df['警戒日'] = df['日'].apply(lambda x: 1 if 10 <= x <= 20 else 0)
corner_list = [521, 540, 541, 560]
df['角台'] = df['台番号'].isin(corner_list).astype(int)

for i in range(1, 8):
    df[f'{i}日前の差枚'] = df.groupby('台番号')['差枚'].shift(i).fillna(0)

df['1日前のRB確率'] = df.groupby('台番号')['RB確率'].shift(1).fillna(0.0)
df['7日間合計'] = df['差枚'] + df['1日前の差枚'] + df['2日前の差枚'] + df['3日前の差枚'] + df['4日前の差枚'] + df['5日前の差枚'] + df['6日前の差枚']
df['V字回復候補'] = df['1日前の差枚'].apply(lambda x: 1 if -4000 <= x <= -2500 else 0)
df['回収トラップ'] = df['1日前の差枚'].apply(lambda x: 1 if x > 3000 else 0)

df['翌日の差枚'] = df.groupby('台番号')['差枚'].shift(-1)
df['翌日勝つか'] = (df['翌日の差枚'] > 0).astype(int)

train_df = df.dropna(subset=['翌日の差枚'])
features = ['G数', '差枚', 'BB', 'RB', '合成確率', '還元日', '警戒日', '角台', 'V字回復候補', '回収トラップ',
            '1日前の差枚', '2日前の差枚', '3日前の差枚', '4日前の差枚', '5日前の差枚', '6日前の差枚', '7日前の差枚']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_df[features], train_df['翌日勝つか'])

# ==========================================
# 危険台除外
# ==========================================
latest_df = df[df['日付'] == latest_date].copy()
exclude_condition = (
    (latest_df['差枚'] >= 1000) |
    (latest_df['1日前の差枚'] >= 1000) |
    (latest_df['2日前の差枚'] >= 1000) |
    ((latest_df['RB確率'] > 0) & (latest_df['RB確率'] <= 310.0)) |
    ((latest_df['1日前のRB確率'] > 0) & (latest_df['1日前のRB確率'] <= 310.0))
)
safe_latest_df = latest_df[~exclude_condition].copy()

# ==========================================
# 3. AI予測セクション
# ==========================================
safe_latest_df['明日勝つ確率(%)'] = model.predict_proba(safe_latest_df[features])[:, 1] * 100
recommendations = safe_latest_df.sort_values('明日勝つ確率(%)', ascending=False)
recommendations = recommendations[recommendations['7日間合計'] <= target_7day_max]
recommendations = recommendations.head(top_n_picks)

st.markdown(f"## 🤖 AI予測ランキング")

col_date, col_count = st.columns(2)
with col_date:
    st.markdown(
        f"<div style='background:#1a2a4a; border:1px solid #4a9eff; border-radius:10px; padding:0.8rem 1rem;'>"
        f"<span style='color:#aaa; font-size:0.85rem;'>📅 予測基準日</span><br>"
        f"<span style='color:#4a9eff; font-size:1.3rem; font-weight:bold;'>{latest_date.strftime('%Y年%m月%d日')}</span>"
        f"</div>",
        unsafe_allow_html=True
    )
with col_count:
    st.markdown(
        f"<div style='background:#1a3a2a; border:1px solid #00e676; border-radius:10px; padding:0.8rem 1rem;'>"
        f"<span style='color:#aaa; font-size:0.85rem;'>🎯 ピックアップ台数</span><br>"
        f"<span style='color:#00e676; font-size:1.3rem; font-weight:bold;'>{len(recommendations)} 台</span>"
        f"</div>",
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)
st.info(f"💡 7日計 {target_7day_max}枚以下の安全な台の中から、AI勝率上位 **{len(recommendations)}台** を表示しています。")

result_display = recommendations[['台番号', '明日勝つ確率(%)', '7日間合計', '差枚', 'RB確率', 'BB', 'RB']].copy()
st.dataframe(
    result_display.rename(columns={'明日勝つ確率(%)': '勝率%', '7日間合計': '7日計'}),
    use_container_width=True,
    hide_index=True
)

# ==========================================
# 4. パターンマッチング予測
# ==========================================
high_setting_days = df[(df['RB確率'] > 0) & (df['RB確率'] <= 290) & (df['差枚'] >= 1000)].copy()

st.markdown("---")
st.markdown("## 🎯 パターンマッチング予測")
st.markdown(
    "<p style='color:#aaaaaa;'>過去の爆発台と「波の形・揺れ幅・プラスマイナス域」が一致する台を抽出します。</p>",
    unsafe_allow_html=True
)

if not high_setting_days.empty:
    historical_patterns = []
    for idx, row in high_setting_days.iterrows():
        hw = row[['7日前の差枚', '6日前の差枚', '5日前の差枚', '4日前の差枚', '3日前の差枚', '2日前の差枚', '1日前の差枚']].values.astype(float)
        if np.std(hw) > 0:
            historical_patterns.append({
                'date': row['日付'].strftime('%m/%d'),
                'raw_date': row['日付'],
                'machine': row['台番号'],
                'wave': hw,
                'depth': np.max(hw) - np.min(hw),
                'bottom': np.min(hw)
            })

    if historical_patterns:
        match_results = []
        pm_candidates = safe_latest_df[safe_latest_df['7日間合計'] <= target_7day_max]

        for idx, row in pm_candidates.iterrows():
            current_machine = row['台番号']
            cw_diffs = [
                row['6日前の差枚'], row['5日前の差枚'], row['4日前の差枚'],
                row['3日前の差枚'], row['2日前の差枚'], row['1日前の差枚'], row['差枚']
            ]
            cw = np.array(cw_diffs, dtype=float)

            if np.std(cw) > 0:
                cw_depth = np.max(cw) - np.min(cw)
                cw_bottom = np.min(cw)
                best_match_score = -1
                best_match_info = ""
                best_match_date = None
                best_match_machine = None

                for hp in historical_patterns:
                    score = np.corrcoef(cw, hp['wave'])[0, 1]
                    depth_diff = abs(cw_depth - hp['depth'])
                    bottom_diff = abs(cw_bottom - hp['bottom'])

                    if not np.isnan(score) and score > best_match_score and depth_diff <= 1000 and bottom_diff <= 1000:
                        best_match_score = score
                        best_match_info = f"{hp['date']}の{hp['machine']}番台"
                        best_match_date = hp['raw_date']
                        best_match_machine = hp['machine']

                if best_match_score >= (pattern_strictness / 100.0):
                    match_results.append({
                        '台番号': current_machine,
                        '類似度(%)': round(best_match_score * 100, 1),
                        '一致した過去の爆発台': best_match_info,
                        '現在の7日計': row['7日間合計'],
                        'past_date': best_match_date,
                        'past_machine': best_match_machine,
                        'cw_diffs': cw_diffs
                    })

        match_df = pd.DataFrame(match_results)
        if not match_df.empty:
            match_df_sorted = match_df.sort_values('類似度(%)', ascending=False)
            st.success(f"🔥 波形が一致する台を **{len(match_df_sorted)}台** 発見しました！")
            st.dataframe(
                match_df_sorted[['台番号', '類似度(%)', '一致した過去の爆発台', '現在の7日計']],
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### 📈 波形比較グラフ")

            for idx, m_row in match_df_sorted.iterrows():
                current_machine = m_row['台番号']
                p_mach = m_row['past_machine']
                p_date = m_row['past_date']
                cw_diffs = m_row['cw_diffs']

                cw_cum = [0]
                cw_daily = [0]
                c_sum = 0
                for d in cw_diffs:
                    c_sum += d
                    cw_cum.append(c_sum)
                    cw_daily.append(d)
                cw_cum.extend([None, None])
                cw_daily.extend([None, None])

                p_history = df[df['台番号'] == p_mach].sort_values('日付').reset_index(drop=True)
                target_idx_list = p_history[p_history['日付'] == p_date].index

                if len(target_idx_list) > 0:
                    t_idx = target_idx_list[0]
                    target_record = p_history.loc[t_idx]

                    pw_diffs = [
                        target_record['7日前の差枚'], target_record['6日前の差枚'],
                        target_record['5日前の差枚'], target_record['4日前の差枚'],
                        target_record['3日前の差枚'], target_record['2日前の差枚'],
                        target_record['1日前の差枚']
                    ]

                    pw_cum = [0]
                    pw_daily = [0]
                    p_sum = 0
                    for d in pw_diffs:
                        p_sum += d
                        pw_cum.append(p_sum)
                        pw_daily.append(d)

                    t_diff = target_record['差枚']
                    p_sum += t_diff
                    pw_cum.append(p_sum)
                    pw_daily.append(t_diff)

                    if t_idx + 1 < len(p_history):
                        next_record = p_history.loc[t_idx + 1]
                        n_diff = next_record['差枚']
                        p_sum += n_diff
                        pw_cum.append(p_sum)
                        pw_daily.append(n_diff)
                    else:
                        pw_cum.append(None)
                        pw_daily.append(None)

                    x_labels = ["起点", "6日前", "5日前", "4日前", "3日前", "2日前", "1日前", "現在(前日)", "★爆発", "🚀翌日"]
                    past_label = f"過去: {p_mach}番台"
                    curr_label = f"現在: {current_machine}番台"

                    plot_data = []
                    for i, label in enumerate(x_labels):
                        if pw_cum[i] is not None:
                            plot_data.append({'期間': label, '種別': past_label, '累積差枚': pw_cum[i], '日別差枚': pw_daily[i]})
                        if cw_cum[i] is not None:
                            plot_data.append({'期間': label, '種別': curr_label, '累積差枚': cw_cum[i], '日別差枚': cw_daily[i]})

                    df_plot = pd.DataFrame(plot_data)

                    with st.expander(f"📊 【現在 {current_machine}番台】 ➡️ 【過去 {p_mach}番台】と比較　類似度: {m_row['類似度(%)']:.1f}%", expanded=True):

                        st.markdown(f"**▼ 過去の爆発当日（{p_date.strftime('%m/%d')}）の詳細データ**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("差枚", f"{int(target_record['差枚']):+}枚")
                        with col2:
                            st.metric("BB回数", f"{int(target_record['BB'])}回")
                        with col3:
                            st.metric("RB回数", f"{int(target_record['RB'])}回")
                        with col4:
                            st.metric("合成確率", str(target_record['合成確率_表示用']))

                        st.markdown("---")

                        base = alt.Chart(df_plot).encode(
                            x=alt.X('期間:O', sort=x_labels, title="差枚数 (棒=日別 / 線=累積)", axis=alt.Axis(labelAngle=0, titlePadding=10, labelColor="#cccccc", titleColor="#aaaaaa")),
                            color=alt.Color('種別:N', legend=alt.Legend(title="", orient="top"),
                                scale=alt.Scale(range=["#f0a500", "#4a9eff"]))
                        )

                        bars = base.mark_bar(opacity=0.6).encode(
                            xOffset='種別:N',
                            y=alt.Y('日別差枚:Q', title="", axis=alt.Axis(minExtent=45, labelColor="#cccccc"))
                        )

                        lines = base.mark_line(size=3).encode(y=alt.Y('累積差枚:Q'))
                        points = base.mark_circle(size=60, opacity=1).encode(y=alt.Y('累積差枚:Q'))

                        chart = alt.layer(bars, lines, points).resolve_scale(y='shared').properties(
                            height=350,
                            background="transparent"
                        ).configure_view(
                            strokeOpacity=0
                        )

                        st.altair_chart(chart, use_container_width=True)
        else:
            st.info(f"現在、波形一致度が {pattern_strictness}% を超える台はありませんでした。")
