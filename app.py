import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time

# ==========================================
# 画面の基本設定
# ==========================================
st.set_page_config(page_title="マイジャグラー5 予測AI", layout="wide")
st.title("🎰 マイジャグラー5 狙い目予測AI")
st.write("過去データとアナリスト理論に基づき、明日の勝率が高い台をランキング表示します。")

# ==========================================
# 1. データの読み込み（セキュリティ強化版）
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
    
    # 前処理（数値変換）
    df['G数'] = pd.to_numeric(df['G数'].astype(str).str.replace(",", ""), errors='coerce').fillna(0).astype(int)
    df['差枚'] = pd.to_numeric(df['差枚'].astype(str).str.replace("+", "").str.replace(",", ""), errors='coerce').fillna(0).astype(int)
    df['BB'] = pd.to_numeric(df['BB'], errors='coerce').fillna(0).astype(int)
    df['RB'] = pd.to_numeric(df['RB'], errors='coerce').fillna(0).astype(int)
    
    df['合成確率_表示用'] = df['合成確率'].astype(str)
    
    # 確率データの分母だけを抽出（例：1/290.0 → 290.0）
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

# 過去7日間の履歴
for i in range(1, 8):
    df[f'{i}日前の差枚'] = df.groupby('台番号')['差枚'].shift(i).fillna(0)

# 1日前、2日前のボーナス情報
df['1日前のBB'] = df.groupby('台番号')['BB'].shift(1).fillna(0).astype(int)
df['1日前のRB'] = df.groupby('台番号')['RB'].shift(1).fillna(0).astype(int)
df['1日前の合成_表示用'] = df.groupby('台番号')['合成確率_表示用'].shift(1).fillna('-')

# 🌟 1日前のRB確率も計算して持たせておく（除外判定に使うため）
df['1日前のRB確率'] = df.groupby('台番号')['RB確率'].shift(1).fillna(0.0)

df['2日前のBB'] = df.groupby('台番号')['BB'].shift(2).fillna(0).astype(int)
df['2日前のRB'] = df.groupby('台番号')['RB'].shift(2).fillna(0).astype(int)
df['2日前の合成_表示用'] = df.groupby('台番号')['合成確率_表示用'].shift(2).fillna('-')

# 7日間合計（本日＋過去6日）
df['7日間合計'] = df['差枚'] + df['1日前の差枚'] + df['2日前の差枚'] + df['3日前の差枚'] + df['4日前の差枚'] + df['5日前の差枚'] + df['6日前の差枚']

df['V字回復候補'] = df['1日前の差枚'].apply(lambda x: 1 if -4000 <= x <= -2500 else 0)
df['回収トラップ'] = df['1日前の差枚'].apply(lambda x: 1 if x > 3000 else 0)

df['翌日の差枚'] = df.groupby('台番号')['差枚'].shift(-1)
df['翌日勝つか'] = (df['翌日の差枚'] > 0).astype(int)

train_df = df.dropna(subset=['翌日の差枚'])
features = ['G数', '差枚', 'BB', 'RB', '合成確率', '還元日', '警戒日', '角台', 'V字回復候補', '回収トラップ',
            '1日前の差枚', '2日前の差枚', '3日前の差枚', '4日前の差枚', '5日前の差枚', '6日前の差枚', '7日前の差枚']

X_train = train_df[features]
y_train = train_df['翌日勝つか']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 3. 予測と表示
# ==========================================
latest_df = df[df['日付'] == latest_date].copy()
latest_df['明日勝つ確率(%)'] = model.predict_proba(latest_df[features])[:, 1] * 100
recommendations = latest_df.sort_values('明日勝つ確率(%)', ascending=False)

# 🌟【足切りルール強化】
# 1. 直近3日間で+1000以上、または7日間合計+2000以上を除外
# 2. 【新規追加】本日または前日のRB確率が設定4（1/290.0）より良い台を除外
exclude_condition = (
    (recommendations['差枚'] >= 1000) | 
    (recommendations['1日前の差枚'] >= 1000) | 
    (recommendations['2日前の差枚'] >= 1000) | 
    (recommendations['7日間合計'] >= 2000) |
    ((recommendations['RB確率'] > 0) & (recommendations['RB確率'] <= 290.0)) |
    ((recommendations['1日前のRB確率'] > 0) & (recommendations['1日前のRB確率'] <= 290.0))
)
recommendations = recommendations[~exclude_condition]

# テーブル表示用のデータ整形
result_display = recommendations[[
    '台番号', '明日勝つ確率(%)', '7日間合計', 
    'BB', 'RB', '合成確率_表示用', '差枚', 
    '1日前のBB', '1日前のRB', '1日前の合成_表示用', '1日前の差枚', 
    '2日前のBB', '2日前のRB', '2日前の合成_表示用', '2日前の差枚'
]].head(7).copy()

result_display = result_display.rename(columns={
    '明日勝つ確率(%)': '勝率%', '7日間合計': '7日計', '合成確率_表示用': '合成',
    '1日前のBB': '前BB', '1日前のRB': '前RB', '1日前の合成_表示用': '前合成', '1日前の差枚': '前差',
    '2日前のBB': '前々BB', '2日前のRB': '前々RB', '2日前の合成_表示用': '前々合成', '2日前の差枚': '前々差'
})
result_display['勝率%'] = result_display['勝率%'].round(1)

# メイン画面表示
st.subheader(f"📅 予測基準日: {latest_date.strftime('%Y-%m-%d')}")
st.info("💡 【安全運用中】直近3日間（当日・前日・前々日）で+1000枚以上、または過去7日間合計が+2000枚以上の台を除外しています。\n\n⚠️ 【高設定不発狙い除外】本日または前日のRB確率が設定4（1/290.0）より良い高設定挙動の台もリスク回避のため除外対象としています。")

st.dataframe(result_display, use_container_width=True, hide_index=True)

# 🌟 個別台の7日間推移グラフ（累積差枚追加）
st.markdown("---")
st.subheader("📈 オススメ台の7日間トレンド（波）")

date_labels = [
    (latest_date - pd.Timedelta(days=6)).strftime('%m/%d'),
    (latest_date - pd.Timedelta(days=5)).strftime('%m/%d'),
    (latest_date - pd.Timedelta(days=4)).strftime('%m/%d'),
    (latest_date - pd.Timedelta(days=3)).strftime('%m/%d'),
    (latest_date - pd.Timedelta(days=2)).strftime('%m/%d'),
    (latest_date - pd.Timedelta(days=1)).strftime('%m/%d'),
    latest_date.strftime('%m/%d(本日)')
]

for index, row in result_display.iterrows():
    machine_no = row['台番号']
    target_machine_data = recommendations[recommendations['台番号'] == machine_no].iloc[0]
    
    history_diff = [
        target_machine_data['6日前の差枚'],
        target_machine_data['5日前の差枚'],
        target_machine_data['4日前の差枚'],
        target_machine_data['3日前の差枚'],
        target_machine_data['2日前の差枚'],
        target_machine_data['1日前の差枚'],
        target_machine_data['差枚']
    ]
    
    # データフレーム作成（まずは日別差枚）
    chart_data = pd.DataFrame({
        '日別差枚': history_diff
    }, index=date_labels)
    
    # 🌟 累積差枚（スランプグラフ用）を計算して追加
    chart_data['累積差枚'] = chart_data['日別差枚'].cumsum()
    
    with st.expander(f"📊 台番号 {machine_no} の詳細トレンドを表示", expanded=True):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("予測勝率", f"{row['勝率%']}%")
            st.metric("7日間合計", f"{row['7日計']}枚")
        with col2:
            st.line_chart(chart_data)

st.caption("※「日別差枚」はその日の単独の差枚、「累積差枚」は6日前から足し算していったスランプグラフです。")