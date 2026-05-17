import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time

# ==========================================
# 画面の基本設定 & サイドバー（絞り込み設定）
# ==========================================
st.set_page_config(page_title="マイジャグラー5 予測AI", layout="wide")

st.sidebar.header("⚙️ 絞り込みフィルター")
st.sidebar.write("数値を動かすと、リアルタイムで結果が絞り込まれます。")

# ユーザーが自由に動かせるスライダー
target_7day_max = st.sidebar.slider("① 7日計の上限 (凹み台狙い)", min_value=-5000, max_value=2000, value=0, step=100)
top_n_picks = st.sidebar.slider("② ピックアップ台数", min_value=1, max_value=15, value=5, step=1)
pattern_strictness = st.sidebar.slider("③ 波形の一致度 (高いほど厳密)", min_value=70, max_value=99, value=90, step=1)

st.title("🎰 マイジャグラー5 狙い目予測AI & 傾向分析")
st.write("過去データに基づき、明日の勝率予測と高設定投入パターンの分析を行います。")

# ==========================================
# 1. データの読み込み
# ==========================================
@st.cache_data(ttl=600) 
def load_data():
    cache_buster = int(time.time())
    try:
        base_url = st.secrets["spreadsheet_url"]
    except KeyError:
        st.error("❌ セキュリティ用の設定が見つかりません。")
        st.stop()
        
    url = f"{base_url}&_={cache_buster}"
    df = pd.read_csv(url)
    
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
    df = load_data()
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
# 3. メイン画面：AIによる明日の予測
# ==========================================
latest_df = df[df['日付'] == latest_date].copy()
latest_df['明日勝つ確率(%)'] = model.predict_proba(latest_df[features])[:, 1] * 100
recommendations = latest_df.sort_values('明日勝つ確率(%)', ascending=False)

# 危険な台の除外（ここは固定）
exclude_condition = (
    (recommendations['差枚'] >= 1000) | 
    (recommendations['1日前の差枚'] >= 1000) | 
    (recommendations['2日前の差枚'] >= 1000) | 
    ((recommendations['RB確率'] > 0) & (recommendations['RB確率'] <= 310.0)) |
    ((recommendations['1日前のRB確率'] > 0) & (recommendations['1日前のRB確率'] <= 310.0))
)
recommendations = recommendations[~exclude_condition]

# 🌟 サイドバーで設定した条件で絞り込み
recommendations = recommendations[recommendations['7日間合計'] <= target_7day_max]

# 勝率の絶対値ではなく、残った中からトップN台を抽出
recommendations = recommendations.head(top_n_picks)

st.subheader(f"📅 予測基準日: {latest_date.strftime('%Y-%m-%d')}")
st.info(f"💡 【AI予測】条件（7日計 {target_7day_max}枚以下）をクリアした台の中から、AI勝率上位 **{len(recommendations)}台** を表示しています。")

result_display = recommendations[['台番号', '明日勝つ確率(%)', '7日間合計', '差枚', 'RB確率', 'BB', 'RB']].copy()
st.dataframe(result_display.rename(columns={'明日勝つ確率(%)': '勝率%', '7日間合計': '7日計'}), use_container_width=True, hide_index=True)

# ==========================================
# 4. パターンマッチング予測 (波の形 + 深さで抽出)
# ==========================================
st.markdown("---")
st.subheader("🎯 パターンマッチング予測 (波の形＋深さ)")

high_setting_days = df[(df['RB確率'] > 0) & (df['RB確率'] <= 290) & (df['差枚'] >= 1000)].copy()

if not high_setting_days.empty:
    historical_patterns = []
    for idx, row in high_setting_days.iterrows():
        hw = row[['7日前の差枚', '6日前の差枚', '5日前の差枚', '4日前の差枚', '3日前の差枚', '2日前の差枚', '1日前の差枚']].values.astype(float)
        if np.std(hw) > 0:
            # 過去の波の「深さ（最大値と最小値の差）」を記録
            hw_depth = np.max(hw) - np.min(hw)
            historical_patterns.append({
                'date': row['日付'].strftime('%m/%d'),
                'machine': row['台番号'],
                'wave': hw,
                'depth': hw_depth
            })

    if historical_patterns:
        match_results = []
        for idx, row in latest_df.iterrows():
            current_machine = row['台番号']
            cw = row[['6日前の差枚', '5日前の差枚', '4日前の差枚', '3日前の差枚', '2日前の差枚', '1日前の差枚', '差枚']].values.astype(float)

            if np.std(cw) > 0:
                cw_depth = np.max(cw) - np.min(cw)
                best_match_score = -1
                best_match_info = ""

                for hp in historical_patterns:
                    # 1. 形が似ているか（相関係数）
                    score = np.corrcoef(cw, hp['wave'])[0, 1]
                    # 2. 波の規模（深さ）が過去の爆発前と近いか（誤差1000枚以内）
                    depth_diff = abs(cw_depth - hp['depth'])
                    
                    if not np.isnan(score) and score > best_match_score and depth_diff <= 1000:
                        best_match_score = score
                        best_match_info = f"{hp['date']}の{hp['machine']}番台"

                # 🌟 サイドバーで設定した一致度（%）以上なら抽出
                if best_match_score >= (pattern_strictness / 100.0):
                    match_results.append({
                        '台番号': current_machine,
                        '類似度(%)': round(best_match_score * 100, 1),
                        '一致した過去の爆発台': best_match_info,
                        '現在の7日計': row['7日間合計']
                    })

        match_df = pd.DataFrame(match_results)
        if not match_df.empty:
            match_df = match_df.sort_values('類似度(%)', ascending=False)
            st.success(f"🔥 過去の爆発前と「波の形」も「深さ」もそっくりな台を **{len(match_df)}台** 発見しました！")
            st.dataframe(match_df, use_container_width=True, hide_index=True)
        else:
            st.info(f"現在、波形一致度が {pattern_strictness}% を超える台はありませんでした。サイドバーから一致度を下げるか、明日は慎重な立ち回りを推奨します。")