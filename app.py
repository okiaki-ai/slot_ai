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
# カスタムCSS（ライト版1デザイン）
# ==========================================
st.markdown("""
<style>
/* 全体背景 */
.stApp {
    background-color: #f8f8fc;
    color: #1a1a2e;
}

/* サイドバー */
[data-testid="stSidebar"] {
    background: #f0eeff;
    border-right: 1px solid #e0dff5;
}
[data-testid="stSidebar"] * { color: #1a1a2e !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #5a4fcf !important; }

/* タイトル */
h1 {
    color: #5a4fcf !important;
    font-weight: 700 !important;
    text-align: center;
}
h2, h3 { color: #5a4fcf !important; }

/* infoボックス */
[data-testid="stInfo"] {
    background: #f0eeff !important;
    border-left: 4px solid #7c6fe0 !important;
    border-radius: 8px;
}

/* successボックス */
[data-testid="stSuccess"] {
    background: #eafaf1 !important;
    border-left: 4px solid #00a85a !important;
    border-radius: 8px;
}

/* dataframe（テーブル背景を白で統一） */
[data-testid="stDataFrame"] {
    border: 1px solid #e0dff5 !important;
    border-radius: 10px;
    overflow: hidden;
    background: #fff !important;
}
[data-testid="stDataFrame"] * {
    background: #fff !important;
    color: #1a1a2e !important;
}

/* メトリクス */
[data-testid="stMetric"] {
    background: #fff;
    border: 0.5px solid #e0dff5;
    border-left: 3px solid #7c6fe0;
    border-radius: 8px;
    padding: 0.8rem 1rem;
}
[data-testid="stMetricLabel"] { color: #999 !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #5a4fcf !important; font-weight: 600 !important; }

/* expander */
[data-testid="stExpander"] {
    background: #fff !important;
    border: 0.5px solid #e0dff5 !important;
    border-radius: 10px !important;
}

/* セパレーター */
hr { border-color: #e0dff5 !important; }

/* selectbox */
[data-testid="stSelectbox"] label { color: #5a4fcf !important; font-weight: 600; }

/* ランキングカードCSS */
.rank-card {
    background: #fff;
    border: 0.5px solid #e0dff5;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: box-shadow 0.15s;
}
.rank-card:hover { box-shadow: 0 2px 8px rgba(124,111,224,0.15); }
.rank-card.top1 { border-color: #7c6fe0; border-left: 3px solid #7c6fe0; }
.rank-card-header { display: flex; align-items: center; justify-content: space-between; }
.rank-left { display: flex; align-items: center; gap: 10px; }
.rank-circle {
    width: 22px; height: 22px; border-radius: 50%;
    background: #f0eeff; display: flex; align-items: center;
    justify-content: center; font-size: 11px; font-weight: 600; color: #aaa;
    flex-shrink: 0;
}
.rank-circle.gold { background: #fff5e0; color: #c88a00; }
.rank-name { font-size: 14px; font-weight: 600; color: #1a1a2e; }
.rank-meta { font-size: 11px; color: #aaa; margin-top: 2px; }
.rank-right { display: flex; align-items: center; gap: 8px; }
.rank-pct { font-size: 15px; font-weight: 700; color: #00a85a; }
.rank-arrow { font-size: 16px; color: #ccc; }
.val-pos { color: #00a85a; font-weight: 600; }
.val-neg { color: #e03e3e; font-weight: 600; }

/* マッチングカード */
.match-card {
    background: #fff;
    border: 0.5px solid #e0dff5;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: box-shadow 0.15s;
}
.match-card:hover { box-shadow: 0 2px 8px rgba(124,111,224,0.15); }
.match-name { font-size: 14px; font-weight: 600; color: #1a1a2e; }
.match-desc { font-size: 11px; color: #aaa; margin-top: 2px; }
.match-score { font-size: 15px; font-weight: 700; color: #00a85a; }

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
# セッションステート初期化
# ==========================================
if 'page' not in st.session_state:
    st.session_state.page = 'list'
if 'selected_machine_data' not in st.session_state:
    st.session_state.selected_machine_data = None
if 'detail_type' not in st.session_state:
    st.session_state.detail_type = None

# ==========================================
# サイドバー（詳細ページでは非表示）
# ==========================================
if st.session_state.page == 'list':
    st.sidebar.markdown("## ⚙️ 絞り込みフィルター")
    st.sidebar.write("数値を動かすと結果がリアルタイムで絞り込まれます。")
    st.sidebar.markdown("---")

    MACHINE_SHEET_MAP = {
        "🎰 マイジャグラー5": "maijag5",
        "🎰 ゴーゴージャグラー": "gojag",
        "🎰 ハッピージャグラー": "happy",
    }
    selected_machine_label = st.sidebar.selectbox(
        "🕹️ 機種を選択", list(MACHINE_SHEET_MAP.keys())
    )
    selected_sheet = MACHINE_SHEET_MAP[selected_machine_label]
    st.session_state.selected_sheet_for_detail = selected_sheet
    st.sidebar.markdown("---")
    target_7day_max = st.sidebar.slider("① 7日計の上限", min_value=-5000, max_value=2000, value=0, step=100)
    top_n_picks = st.sidebar.slider("② ピックアップ台数", min_value=1, max_value=15, value=5, step=1)
    pattern_strictness = st.sidebar.slider("③ 波形の一致度", min_value=70, max_value=99, value=90, step=1)

# ==========================================
# 1. データ読み込み
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

# ==========================================
# 詳細ページ
# ==========================================
def show_detail_page(df):
    d = st.session_state.selected_machine_data
    detail_type = st.session_state.detail_type

    if st.button("← 一覧に戻る"):
        st.session_state.page = 'list'
        st.rerun()

    st.markdown(f"## {d['台番号']}番台 詳細")
    st.markdown("---")

    if detail_type == 'ai':
        # --- 勝率 ---
        st.markdown(
            f"<div style='text-align:center; font-size:2.5rem; font-weight:700; color:#00a85a;'>{d['勝率']:.1f}%</div>"
            f"<div style='text-align:center; font-size:0.85rem; color:#aaa; margin-bottom:1rem;'>明日の勝率予測</div>",
            unsafe_allow_html=True
        )

        # --- BB/RB/合成確率 ---
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("BB回数", f"{d['BB']}回")
        with col2: st.metric("RB回数", f"{d['RB']}回")
        with col3: st.metric("合成確率", f"1/{d['合成確率']:.0f}" if d['合成確率'] > 0 else "-")

        st.markdown("---")

        # --- 過去7日間の差枚推移（上下グラフ）---
        st.markdown("### 📊 過去7日間の差枚推移")
        machine_history = df[df['台番号'] == d['台番号']].sort_values('日付').tail(8)
        if len(machine_history) >= 2:
            recent = machine_history.iloc[-8:-1] if len(machine_history) >= 8 else machine_history.iloc[:-1]
            chart_data = pd.DataFrame({
                '日付': recent['日付'].dt.strftime('%m/%d'),
                '差枚': recent['差枚'],
                '色': recent['差枚'].apply(lambda x: 'プラス' if x >= 0 else 'マイナス')
            })
            bars = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X('日付:O', sort=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y('差枚:Q', title='差枚'),
                color=alt.Color('色:N', scale=alt.Scale(
                    domain=['プラス', 'マイナス'],
                    range=['#00a85a', '#e03e3e']
                ), legend=None)
            ).properties(height=250)
            zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='#aaa', strokeDash=[3,3]).encode(y='y:Q')
            st.altair_chart(alt.layer(bars, zero_line), use_container_width=True)

    elif detail_type == 'pattern':
        # --- 類似度 ---
        st.markdown(
            f"<div style='text-align:center; font-size:2.5rem; font-weight:700; color:#00a85a;'>{d['類似度']:.1f}%</div>"
            f"<div style='text-align:center; font-size:0.85rem; color:#aaa; margin-bottom:1rem;'>パターン一致度</div>",
            unsafe_allow_html=True
        )
        st.info(f"📌 一致した過去の爆発台：{d['一致した過去の爆発台']}")

        # --- 過去の爆発台詳細 ---
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("差枚（爆発日）", f"{d['past_diff']:+}枚")
        with col2: st.metric("BB回数", f"{d['past_BB']}回")
        with col3: st.metric("RB回数", f"{d['past_RB']}回")

        st.markdown("---")

        # --- 波形比較グラフ（棒＋線）---
        st.markdown("### 📈 波形比較（現在 vs 過去の爆発台）")
        x_labels = ["起点","6日前","5日前","4日前","3日前","2日前","1日前","現在(前日)","★爆発","🚀翌日"]

        cw_diffs = d['cw_diffs']
        cw_cum = [0]
        cw_daily = [0]
        c_sum = 0
        for v in cw_diffs:
            c_sum += v
            cw_cum.append(c_sum)
            cw_daily.append(v)
        cw_cum.extend([None, None])
        cw_daily.extend([None, None])

        pw_cum = d['pw_cum']
        pw_daily = d['pw_daily']

        past_label = f"過去: {d['past_machine']}番台"
        curr_label = f"現在: {d['台番号']}番台"

        plot_data = []
        for i, label in enumerate(x_labels):
            if i < len(pw_cum) and pw_cum[i] is not None:
                plot_data.append({'期間': label, '種別': past_label, '累積差枚': pw_cum[i], '日別差枚': pw_daily[i]})
            if i < len(cw_cum) and cw_cum[i] is not None:
                plot_data.append({'期間': label, '種別': curr_label, '累積差枚': cw_cum[i], '日別差枚': cw_daily[i]})

        df_plot = pd.DataFrame(plot_data)

        base = alt.Chart(df_plot).encode(
            x=alt.X('期間:O', sort=x_labels, axis=alt.Axis(labelAngle=-30, titlePadding=10)),
            color=alt.Color('種別:N', legend=alt.Legend(title="", orient="top"),
                scale=alt.Scale(domain=[past_label, curr_label], range=["#7c6fe0", "#00a85a"]))
        )
        bars = base.mark_bar(opacity=0.5).encode(
            xOffset='種別:N',
            y=alt.Y('日別差枚:Q', title="日別差枚")
        )
        lines = base.mark_line(size=2.5).encode(y=alt.Y('累積差枚:Q', title="累積差枚"))
        points = base.mark_circle(size=50).encode(y=alt.Y('累積差枚:Q'))
        chart = alt.layer(bars, lines, points).resolve_scale(y='independent').properties(height=300)
        st.altair_chart(chart, use_container_width=True)

# ==========================================
# 一覧ページ
# ==========================================
def show_list_page():
    with st.spinner(f'データを読み込み中...'):
        df = load_data(selected_sheet)
        latest_date = df['日付'].max()

    # AI学習
    df['日'] = df['日付'].dt.day
    df['還元日'] = df['日'].apply(lambda x: 1 if x == 3 or (1 <= x <= 5) or (27 <= x <= 31) else 0)
    df['警戒日'] = df['日'].apply(lambda x: 1 if 10 <= x <= 20 else 0)
    corner_list = [521, 540, 541, 560]
    df['角台'] = df['台番号'].isin(corner_list).astype(int)
    for i in range(1, 8):
        df[f'{i}日前の差枚'] = df.groupby('台番号')['差枚'].shift(i).fillna(0)
    df['1日前のRB確率'] = df.groupby('台番号')['RB確率'].shift(1).fillna(0.0)
    df['7日間合計'] = sum(df[f'{i}日前の差枚'] if i > 0 else df['差枚'] for i in range(8)).fillna(0) if False else (
        df['差枚'] + df['1日前の差枚'] + df['2日前の差枚'] + df['3日前の差枚'] +
        df['4日前の差枚'] + df['5日前の差枚'] + df['6日前の差枚']
    )
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

    # タイトル
    st.markdown(f"# 🎰 {selected_machine_label.replace('🎰 ', '')} 予測AI")
    st.markdown(
        f"<p style='text-align:center; color:#aaa; font-size:0.9rem;'>予測基準日：{latest_date.strftime('%Y年%m月%d日')}</p>",
        unsafe_allow_html=True
    )

    # AI予測ランキング
    st.markdown('<div class="section-label">AI予測ランキング</div>', unsafe_allow_html=True)
    st.info(f"💡 7日計 {target_7day_max}枚以下の台から勝率上位 {len(recommendations)}台 を表示")

    for rank, (_, row) in enumerate(recommendations.iterrows(), 1):
        card_class = "rank-card top1" if rank == 1 else "rank-card"
        circle_class = "rank-circle gold" if rank == 1 else "rank-circle"
        diff_class = "val-pos" if row['差枚'] >= 0 else "val-neg"
        diff_sign = "+" if row['差枚'] >= 0 else ""
        sum_class = "val-pos" if row['7日間合計'] >= 0 else "val-neg"
        sum_sign = "+" if row['7日間合計'] >= 0 else ""

        btn_key = f"ai_{rank}_{row['台番号']}"
        st.markdown(f"""
        <div class="{card_class}">
          <div class="rank-card-header">
            <div class="rank-left">
              <div class="{circle_class}">{rank}</div>
              <div>
                <div class="rank-name">{int(row['台番号'])}番台</div>
                <div class="rank-meta">
                  7日計 <span class="{sum_class}">{sum_sign}{int(row['7日間合計'])}</span>
                  &nbsp; 差枚 <span class="{diff_class}">{diff_sign}{int(row['差枚'])}</span>
                </div>
              </div>
            </div>
            <div class="rank-right">
              <div class="rank-pct">{row['明日勝つ確率(%)']:.1f}%</div>
              <div class="rank-arrow">›</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(f"{int(row['台番号'])}番台の詳細を見る", key=btn_key):
            st.session_state.selected_machine_data = {
                '台番号': int(row['台番号']),
                '勝率': row['明日勝つ確率(%)'],
                'BB': int(row['BB']),
                'RB': int(row['RB']),
                '合成確率': row['合成確率'],
            }
            st.session_state.detail_type = 'ai'
            st.session_state.page = 'detail'
            st.rerun()

    # パターンマッチング
    st.markdown('<div class="section-label">パターンマッチング予測</div>', unsafe_allow_html=True)

    high_setting_days = df[(df['RB確率'] > 0) & (df['RB確率'] <= 290) & (df['差枚'] >= 1000)].copy()

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
                    # 過去の波形データ構築
                    p_history = df[df['台番号'] == best_hp['machine']].sort_values('日付').reset_index(drop=True)
                    t_idx_list = p_history[p_history['日付'] == best_hp['raw_date']].index
                    pw_cum = []
                    pw_daily = []
                    if len(t_idx_list) > 0:
                        t_idx = t_idx_list[0]
                        t_rec = p_history.loc[t_idx]
                        pw_diffs = [t_rec['7日前の差枚'],t_rec['6日前の差枚'],t_rec['5日前の差枚'],
                                    t_rec['4日前の差枚'],t_rec['3日前の差枚'],t_rec['2日前の差枚'],t_rec['1日前の差枚']]
                        pw_cum = [0]
                        pw_daily = [0]
                        p_sum = 0
                        for v in pw_diffs:
                            p_sum += v
                            pw_cum.append(p_sum)
                            pw_daily.append(v)
                        p_sum += t_rec['差枚']
                        pw_cum.append(p_sum)
                        pw_daily.append(t_rec['差枚'])
                        if t_idx + 1 < len(p_history):
                            n_diff = p_history.loc[t_idx + 1, '差枚']
                            p_sum += n_diff
                            pw_cum.append(p_sum)
                            pw_daily.append(n_diff)
                        else:
                            pw_cum.append(None)
                            pw_daily.append(None)

                    match_results.append({
                        '台番号': int(row['台番号']),
                        '類似度': round(best_score * 100, 1),
                        '一致した過去の爆発台': f"{best_hp['date']}の{best_hp['machine']}番台",
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
                sum_class = "val-pos" if m['現在の7日計'] >= 0 else "val-neg"
                sum_sign = "+" if m['現在の7日計'] >= 0 else ""
                st.markdown(f"""
                <div class="match-card">
                  <div>
                    <div class="match-name">{m['台番号']}番台</div>
                    <div class="match-desc">{m['一致した過去の爆発台']} と一致 ／ 7日計 <span class="{sum_class}">{sum_sign}{m['現在の7日計']}</span></div>
                  </div>
                  <div style="display:flex; align-items:center; gap:6px;">
                    <div class="match-score">{m['類似度']}%</div>
                    <div class="rank-arrow">›</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"【パターン】{m['台番号']}番台の詳細", key=f"pm_{m['台番号']}"):
                    st.session_state.selected_machine_data = m
                    st.session_state.detail_type = 'pattern'
                    st.session_state.page = 'detail'
                    st.rerun()
        else:
            st.info(f"波形一致度 {pattern_strictness}% を超える台はありませんでした。")
    else:
        st.info("過去の高設定データが不足しています。")

    return df

# ==========================================
# ページルーティング
# ==========================================
if st.session_state.page == 'list':
    df = show_list_page()
elif st.session_state.page == 'detail':
    # detailページ用にデータ再取得
    MACHINE_SHEET_MAP = {
        "🎰 マイジャグラー5": "maijag5",
        "🎰 ゴーゴージャグラー": "gojag",
        "🎰 ハッピージャグラー": "happy",
    }
    if 'selected_sheet_for_detail' not in st.session_state:
        st.session_state.selected_sheet_for_detail = "maijag5"
    df = load_data(st.session_state.get('selected_sheet_for_detail', 'maijag5'))
    df['日'] = df['日付'].dt.day
    for i in range(1, 8):
        df[f'{i}日前の差枚'] = df.groupby('台番号')['差枚'].shift(i).fillna(0)
    show_detail_page(df)
