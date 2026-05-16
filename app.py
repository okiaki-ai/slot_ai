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
@st.cache_data(ttl=600) # 10分間はデータをキャッシュして高速化
def load_data():
    cache_buster = int(time.time())
    
    # 【セキュリティ対策】隠し箱（Secrets）から安全に読み込む
    try:
        base_url = st.secrets["spreadsheet_url"]
    except KeyError:
        st.error("❌ セキュリティ用の設定（secrets.toml）が見つかりません。'.streamlit/secrets.toml' ファイルが正しく作成されているか確認してください。")
        st.stop()
        
    url = f"{base_url}&_={cache_buster}"
    df = pd.read_csv(url)
    
    # 前処理（文字のお掃除）
    df['G数'] = pd.to_numeric(df['G数'].astype(str).str.replace(",", ""), errors='coerce').fillna(0).astype(int)
    df['差枚'] = pd.to_numeric(df['差枚'].astype(str).str.replace("+", "").str.replace(",", ""), errors='coerce').fillna(0).astype(int)
    df['BB'] = pd.to_numeric(df['BB'], errors='coerce').fillna(0).astype(int)
    df['RB'] = pd.to_numeric(df['RB'], errors='coerce').fillna(0).astype(int)
    
    # 確率データ（表示用文字列）の保管
    df['合成確率_表示用'] = df['合成確率'].astype(str)
    
    df['BB確率'] = pd.to_numeric(df['BB確率'].astype(str).str.replace("1/", ""), errors='coerce').fillna(0.0)
    df['RB確率'] = pd.to_numeric(df['RB確率'].astype(str).str.replace("1/", ""), errors='coerce').fillna(0.0)
    df['合成確率'] = pd.to_numeric(df['合成確率'].astype(str).str.replace("1/", ""), errors='coerce').fillna(0.0)
    
    df['日付'] = pd.to_datetime(df['日付'])
    return df.sort_values(['台番号', '日付'])

with st.spinner('データを安全に読み込み、AIが学習中です...'):
    df = load_data()
    latest_date = df['日付'].max()

# ==========================================
# 2. 🧠 アナリスト理論の組み込み ＆ AI学習
# ==========================================
df['日'] = df['日付'].dt.day
df['還元日'] = df['日'].apply(lambda x: 1 if x == 3 or (1 <= x <= 5) or (27 <= x <= 31) else 0)
df['警戒日'] = df['日'].apply(lambda x: 1 if 10 <= x <= 20 else 0)

corner_list = [521, 540, 541, 560]
df['角台'] = df['台番号'].isin(corner_list).astype(int)

# 過去7日間のトレンドおよび「前日」「前々日」のボーナス情報の読み込み
for i in range(1, 8):
    df[f'{i}日前の差枚'] = df.groupby('台番号')['差枚'].shift(i).fillna(0)

# 1日前（前日）のデータ取得
df['1日前のBB'] = df.groupby('台番号')['BB'].shift(1).fillna(0).astype(int)
df['1日前のRB'] = df.groupby('台番号')['RB'].shift(1).fillna(0).astype(int)
df['1日前の合成_表示用'] = df.groupby('台番号')['合成確率_表示用'].shift(1).fillna('-')

# 2日前（前々日）のデータ取得
df['2日前のBB'] = df.groupby('台番号')['BB'].shift(2).fillna(0).astype(int)
df['2日前のRB'] = df.groupby('台番号')['RB'].shift(2).fillna(0).astype(int)
df['2日前の合成_表示用'] = df.groupby('台番号')['合成確率_表示用'].shift(2).fillna('-')

# 直近7日間の合計差枚を計算（本日＋過去6日）
df['7日間合計'] = df['差枚'] + df['1日前の差枚'] + df['2日前の差枚'] + df['3日前の差枚'] + df['4日前の差枚'] + df['5日前の差枚'] + df['6日前の差枚']

df['V字回復候補'] = df['1日前の差枚'].apply(lambda x: 1 if -4000 <= x <= -2500 else 0)
df['回収トラップ'] = df['1日前の差枚'].apply(lambda x: 1 if x > 3000 else 0)

df['翌日の差枚'] = df.groupby('台番号')['差枚'].shift(-1)
df['翌日勝つか'] = (df['翌日の差枚'] > 0).astype(int)

train_df = df.dropna(subset=['翌日の差枚'])

features = [
    'G数', '差枚', 'BB', 'RB', '合成確率', 
    '還元日', '警戒日', '角台', 'V字回復候補', '回収トラップ',
    '1日前の差枚', '2日前の差枚', '3日前の差枚', 
    '4日前の差枚', '5日前の差枚', '6日前の差枚', '7日前の差枚'
]

X_train = train_df[features]
y_train = train_df['翌日勝つか']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 3. 予測と結果の表示
# ==========================================
latest_df = df[df['日付'] == latest_date].copy()
latest_df['明日勝つ確率(%)'] = model.predict_proba(latest_df[features])[:, 1] * 100

recommendations = latest_df.sort_values('明日勝つ確率(%)', ascending=False)

# 🌟【足切りルール強化】直近（当日・前日）で+1000枚以上、または過去7日間合計が+2000枚以上の台は強制除外
exclude_condition = (
    (recommendations['差枚'] >= 1000) | 
    (recommendations['1日前の差枚'] >= 1000) | 
    (recommendations['7日間合計'] >= 2000)
)
recommendations = recommendations[~exclude_condition]

# 画面表示用の列を厳選
result = recommendations[[
    '台番号', '明日勝つ確率(%)', '7日間合計', 
    'BB', 'RB', '合成確率_表示用', '差枚', 
    '1日前のBB', '1日前のRB', '1日前の合成_表示用', '1日前の差枚', 
    '2日前のBB', '2日前のRB', '2日前の合成_表示用', '2日前の差枚'
]].head(7)

# スマホでも一覧できるように列の名前を限界まで短縮
result = result.rename(columns={
    '明日勝つ確率(%)': '勝率%',
    '7日間合計': '7日計',
    '合成確率_表示用': '合成',
    '1日前のBB': '前BB',
    '1日前のRB': '前RB',
    '1日前の合成_表示用': '前合成',
    '1日前の差枚': '前差枚',
    '2日前のBB': '前々BB',
    '2日前のRB': '前々RB',
    '2日前の合成_表示用': '前々合成',
    '2日前の差枚': '前々差枚'
})

# パーセント表示の丸め
result['勝率%'] = result['勝率%'].round(1)

# 表示
st.subheader(f"📅 予測基準日: {latest_date.strftime('%Y-%m-%d')}")
st.info("💡 【安全運用中】直近（当日・前日）で+1000枚以上、または過去7日間合計（7日計）が+2000枚以上の台はリスク回避のため強制除外しています。")

# アプリ画面に綺麗な表を描画
st.dataframe(result, use_container_width=True, hide_index=True)

st.caption("※「7日計」は予測基準日を含む直近7日間の合計差枚です。")
st.caption("※「前〜」は1日前（前日）、「前々〜」は2日前（前々日）のデータです。")